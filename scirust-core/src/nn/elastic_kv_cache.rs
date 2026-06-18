//! **Elastic compressed KV-cache** — the shared deterministic primitive behind the
//! SLHAv2 (KV-cache compression so LLMs run in CPU cache, not on an expensive GPU)
//! and CCOS (bounded-memory paging) projects, built on scirust's quantization and
//! determinism.
//!
//! An attention key/value pair is compressed into a [`KvTile`] by **two-level INT4**
//! quantization: a symmetric INT4 base plus an INT4 **residual** (SLHAv2's residual
//! tracking), each with **per-group adaptive scales** (a finer scale per channel
//! group, so groups of very different magnitude are all resolved — the cosine-aware
//! / adaptive scaling SLHAv2 uses, akin to KVQuant's per-channel keys). That lifts
//! the cosine fidelity past 0.99 while shrinking the footprint several-fold versus
//! `f32`. The [`ElasticKvCache`] holds those tiles under an optional **budget** and
//! evicts the oldest when over it (soft-paging / elastic memory), and serves
//! attention straight from the compressed tiles — reusing [`contiguous_attention`]
//! so the only difference from a full-precision cache is the (small, measured)
//! compression error.
//!
//! Everything is pure, **deterministic** `f32`/`i8` arithmetic (no RNG, fixed order),
//! so eviction decisions and attention outputs are bit-for-bit reproducible.

use crate::nn::paged_attention::contiguous_attention;
use std::collections::VecDeque;

/// INT4 magnitude limit (symmetric range `[-7, 7]`).
const QMAX_INT4: f32 = 7.0;

/// Symmetric **INT4** quantization of a vector: per-vector absmax scale, codes in
/// `[-7, 7]`. Returns the codes and the scale (deterministic).
pub fn quantize_int4(x: &[f32]) -> (Vec<i8>, f32) {
    let maxabs = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let scale = if maxabs == 0.0
    {
        1.0
    }
    else
    {
        maxabs / QMAX_INT4
    };
    let codes = x
        .iter()
        .map(|&v| (v / scale).round().clamp(-QMAX_INT4, QMAX_INT4) as i8)
        .collect();
    (codes, scale)
}

/// Reconstruct a vector from INT4 codes and a scale (`codeᵢ · scale`). Runs through
/// the SIMD-accelerated [`scirust_simd::ops::dequantize_int4_into`] kernel, which is
/// **bit-identical** to the scalar form (element-wise, no reduction), so the codec is
/// faster yet stays deterministic across platforms.
pub fn dequantize_int4(codes: &[i8], scale: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; codes.len()];
    scirust_simd::ops::dequantize_int4_into(codes, scale, &mut out);
    out
}

/// **Cosine-aware** grouped INT4 quantization: split `x` into chunks of `group_size`
/// and give each its **own** absmax scale, so a low-magnitude group is not crushed by
/// a high-magnitude one (adaptive scaling). `group_size = x.len()` reduces to a single
/// scale ([`quantize_int4`]). Returns the codes and one scale per group.
pub fn quantize_int4_grouped(x: &[f32], group_size: usize) -> (Vec<i8>, Vec<f32>) {
    let g = group_size.clamp(1, x.len().max(1));
    let mut codes = Vec::with_capacity(x.len());
    let mut scales = Vec::with_capacity(x.len().div_ceil(g));
    for chunk in x.chunks(g)
    {
        let (c, s) = quantize_int4(chunk);
        codes.extend(c);
        scales.push(s);
    }
    (codes, scales)
}

/// Inverse of [`quantize_int4_grouped`].
pub fn dequantize_int4_grouped(codes: &[i8], scales: &[f32], group_size: usize) -> Vec<f32> {
    let g = group_size.clamp(1, codes.len().max(1));
    let mut out = Vec::with_capacity(codes.len());
    for (chunk, &s) in codes.chunks(g).zip(scales)
    {
        out.extend(dequantize_int4(chunk, s));
    }
    out
}

/// Cosine similarity `⟨a,b⟩ / (‖a‖‖b‖)` — the fidelity metric SLHAv2 reports against
/// full attention (`1.0` for identical vectors).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (&x, &y) in a.iter().zip(b)
    {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0
    {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Two-level grouped INT4 code of `x`: an INT4 base, then an INT4 quantization of the
/// residual `x − dequant(base)`, both with per-group scales.
type TwoLevel = (Vec<i8>, Vec<f32>, Vec<i8>, Vec<f32>);
fn compress_vec(x: &[f32], group_size: usize) -> TwoLevel {
    let (base, base_scales) = quantize_int4_grouped(x, group_size);
    let recon = dequantize_int4_grouped(&base, &base_scales, group_size);
    let residual: Vec<f32> = x.iter().zip(&recon).map(|(&a, &b)| a - b).collect();
    let (res, res_scales) = quantize_int4_grouped(&residual, group_size);
    (base, base_scales, res, res_scales)
}

/// A compressed key/value pair: a **two-level INT4** code (base + residual) per
/// vector, with **per-group scales**. Reconstructs to `base + residual`.
#[derive(Clone)]
pub struct KvTile {
    k_base: Vec<i8>,
    k_base_scales: Vec<f32>,
    k_res: Vec<i8>,
    k_res_scales: Vec<f32>,
    v_base: Vec<i8>,
    v_base_scales: Vec<f32>,
    v_res: Vec<i8>,
    v_res_scales: Vec<f32>,
    group_size: usize,
}

impl KvTile {
    /// Compress a `(key, value)` pair (both length `d`) with a **single** scale per
    /// vector (the simplest tile).
    pub fn compress(k: &[f32], v: &[f32]) -> Self {
        Self::compress_grouped(k, v, k.len().max(1))
    }

    /// Compress with **per-group** adaptive scales of width `group_size` (cosine-aware;
    /// smaller groups = finer scales = higher fidelity, a few more scale bytes).
    pub fn compress_grouped(k: &[f32], v: &[f32], group_size: usize) -> Self {
        assert_eq!(k.len(), v.len(), "KvTile: key/value length mismatch");
        let g = group_size.clamp(1, k.len().max(1));
        let (k_base, k_base_scales, k_res, k_res_scales) = compress_vec(k, g);
        let (v_base, v_base_scales, v_res, v_res_scales) = compress_vec(v, g);
        Self {
            k_base,
            k_base_scales,
            k_res,
            k_res_scales,
            v_base,
            v_base_scales,
            v_res,
            v_res_scales,
            group_size: g,
        }
    }

    /// Reconstruct the key (`base + residual`).
    pub fn key(&self) -> Vec<f32> {
        let b = dequantize_int4_grouped(&self.k_base, &self.k_base_scales, self.group_size);
        let r = dequantize_int4_grouped(&self.k_res, &self.k_res_scales, self.group_size);
        b.iter().zip(&r).map(|(&x, &y)| x + y).collect()
    }

    /// Reconstruct the value (`base + residual`).
    pub fn value(&self) -> Vec<f32> {
        let b = dequantize_int4_grouped(&self.v_base, &self.v_base_scales, self.group_size);
        let r = dequantize_int4_grouped(&self.v_res, &self.v_res_scales, self.group_size);
        b.iter().zip(&r).map(|(&x, &y)| x + y).collect()
    }

    /// Packed compressed footprint in bytes: the INT4 codes (½ byte each) plus the
    /// per-group `f32` scales.
    pub fn packed_bytes(&self) -> usize {
        let nibbles = self.k_base.len() + self.k_res.len() + self.v_base.len() + self.v_res.len();
        let scales = self.k_base_scales.len()
            + self.k_res_scales.len()
            + self.v_base_scales.len()
            + self.v_res_scales.len();
        nibbles.div_ceil(2) + scales * std::mem::size_of::<f32>()
    }
}

/// A bounded, **elastic** cache of compressed [`KvTile`]s. With a non-zero `budget`
/// it keeps at most that many tiles, evicting the oldest (soft-paging) — so memory is
/// bounded regardless of sequence length. Attention is served from the compressed
/// tiles. Deterministic.
pub struct ElasticKvCache {
    d: usize,
    budget: usize,     // max resident tiles (0 = unbounded)
    group_size: usize, // per-group scale width for the tile codec
    tiles: VecDeque<KvTile>,
    evicted: usize,
}

impl ElasticKvCache {
    /// New cache for `d`-dimensional keys/values with a single scale per vector.
    /// `budget = 0` means unbounded.
    pub fn new(d: usize, budget: usize) -> Self {
        Self::new_grouped(d, budget, d.max(1))
    }

    /// New cache using **per-group** adaptive scales of width `group_size` (higher
    /// fidelity for heterogeneous channels).
    pub fn new_grouped(d: usize, budget: usize, group_size: usize) -> Self {
        Self {
            d,
            budget,
            group_size: group_size.clamp(1, d.max(1)),
            tiles: VecDeque::new(),
            evicted: 0,
        }
    }

    /// Append a `(key, value)` pair (compressed into a tile); evicts the oldest tile
    /// if that would exceed the budget.
    pub fn append(&mut self, k: &[f32], v: &[f32]) {
        assert_eq!(k.len(), self.d, "append: key length must be d");
        assert_eq!(v.len(), self.d, "append: value length must be d");
        self.tiles
            .push_back(KvTile::compress_grouped(k, v, self.group_size));
        if self.budget != 0 && self.tiles.len() > self.budget
        {
            self.tiles.pop_front();
            self.evicted += 1;
        }
    }

    /// Number of resident tiles.
    pub fn len(&self) -> usize {
        self.tiles.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    /// Number of tiles evicted by the budget so far.
    pub fn evicted(&self) -> usize {
        self.evicted
    }

    /// Total compressed (packed) footprint in bytes.
    pub fn compressed_bytes(&self) -> usize {
        self.tiles.iter().map(KvTile::packed_bytes).sum()
    }

    /// Attention `softmax(q·Kᵀ/√d)·V` over the **compressed** resident tiles, using
    /// the same arithmetic as [`contiguous_attention`] so the only deviation from a
    /// full-precision cache is the compression error.
    pub fn attention(&self, q: &[f32]) -> Vec<f32> {
        let n = self.tiles.len();
        let mut keys = Vec::with_capacity(n * self.d);
        let mut values = Vec::with_capacity(n * self.d);
        for t in &self.tiles
        {
            keys.extend(t.key());
            values.extend(t.value());
        }
        contiguous_attention(&keys, &values, q, self.d, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::PcgEngine;
    use crate::nn::paged_attention::contiguous_attention;

    fn rand_vec(n: usize, rng: &mut PcgEngine) -> Vec<f32> {
        (0..n).map(|_| rng.float_signed()).collect()
    }

    /// A compressed tile reconstructs each vector with **high cosine fidelity**, the
    /// residual level beats the base level alone, and it is deterministic.
    #[test]
    fn tile_high_cosine_fidelity_and_residual_helps() {
        let (d, mut rng) = (128usize, PcgEngine::new(1));
        let k = rand_vec(d, &mut rng);
        let v = rand_vec(d, &mut rng);
        let tile = KvTile::compress(&k, &v);
        let ck = cosine_similarity(&k, &tile.key());
        let cv = cosine_similarity(&v, &tile.value());
        assert!(ck > 0.95 && cv > 0.95, "low fidelity: key {ck}, value {cv}");
        // Residual tracking strictly improves on the base INT4 level alone.
        let (base, bs) = quantize_int4(&k);
        let base_cos = cosine_similarity(&k, &dequantize_int4(&base, bs));
        assert!(
            ck > base_cos,
            "residual did not help: {ck} vs base {base_cos}"
        );
        // Deterministic.
        let tile2 = KvTile::compress(&k, &v);
        assert_eq!(tile.key(), tile2.key());
    }

    /// **Cosine-aware grouped scaling** strictly improves fidelity over a single
    /// scale when channel magnitudes are heterogeneous: a global scale is too coarse
    /// for the smaller (but non-negligible) channels, while per-group scales resolve
    /// each. Shown at the base INT4 level (the primitive); the grouped tile is never
    /// worse than the single-scale tile, at only a few extra scale bytes.
    #[test]
    fn grouped_scaling_improves_fidelity() {
        let d = 128usize;
        let mut rng = PcgEngine::new(5);
        // Moderate, heterogeneous per-group magnitudes (every channel still matters).
        let mut k = vec![0.0f32; d];
        for (i, ki) in k.iter_mut().enumerate()
        {
            let mag = [0.3f32, 0.6, 1.5, 3.0][(i / 32) % 4];
            *ki = mag * rng.float_signed();
        }
        // Base level: grouped scales beat a single global scale.
        let (cs, ss) = quantize_int4(&k);
        let single_base = cosine_similarity(&k, &dequantize_int4(&cs, ss));
        let (cg, sg) = quantize_int4_grouped(&k, 32);
        let grouped_base = cosine_similarity(&k, &dequantize_int4_grouped(&cg, &sg, 32));
        assert!(
            grouped_base > single_base + 1e-3,
            "grouped base {grouped_base} not better than single {single_base}"
        );
        // The full (residual) grouped tile is never worse than the single-scale tile.
        let cos_single = cosine_similarity(&k, &KvTile::compress(&k, &k).key());
        let cos_grouped = cosine_similarity(&k, &KvTile::compress_grouped(&k, &k, 32).key());
        assert!(
            cos_grouped >= cos_single - 1e-6,
            "grouped tile worse: {cos_grouped} vs {cos_single}"
        );
        // The footprint grows only by the extra per-group scales (stays compact).
        assert!(
            KvTile::compress_grouped(&k, &k, 32).packed_bytes()
                <= KvTile::compress(&k, &k).packed_bytes() + 8 * 8 + 16
        );
    }

    /// Attention over the **compressed** cache matches full-precision attention to
    /// high cosine similarity — the end-to-end fidelity SLHAv2 targets.
    #[test]
    fn compressed_attention_matches_full() {
        let (d, n, mut rng) = (64usize, 40usize, PcgEngine::new(2));
        let mut cache = ElasticKvCache::new(d, 0); // unbounded
        let (mut keys, mut values) = (Vec::new(), Vec::new());
        for _ in 0..n
        {
            let k = rand_vec(d, &mut rng);
            let v = rand_vec(d, &mut rng);
            cache.append(&k, &v);
            keys.extend(&k);
            values.extend(&v);
        }
        let q = rand_vec(d, &mut rng);
        let approx = cache.attention(&q);
        let exact = contiguous_attention(&keys, &values, &q, d, n);
        let cos = cosine_similarity(&approx, &exact);
        assert!(cos > 0.99, "compressed attention diverged: cosine {cos}");
    }

    /// The compressed footprint is several-fold smaller than the `f32` key/value.
    #[test]
    fn compression_ratio_is_significant() {
        let (d, n, mut rng) = (128usize, 64usize, PcgEngine::new(3));
        let mut cache = ElasticKvCache::new(d, 0);
        for _ in 0..n
        {
            cache.append(&rand_vec(d, &mut rng), &rand_vec(d, &mut rng));
        }
        let raw = n * 2 * d * std::mem::size_of::<f32>(); // f32 key + value
        let compressed = cache.compressed_bytes();
        assert!(
            compressed * 3 <= raw,
            "weak compression: {compressed} vs raw {raw}"
        );
    }

    /// With a budget the cache stays **bounded** (oldest tiles evicted) and is
    /// **bit-for-bit deterministic** across identical runs.
    #[test]
    fn elastic_budget_is_bounded_and_deterministic() {
        let (d, budget, total) = (32usize, 16usize, 50usize);
        let run = || -> (usize, usize, Vec<f32>) {
            let mut rng = PcgEngine::new(7);
            let mut cache = ElasticKvCache::new(d, budget);
            for _ in 0..total
            {
                cache.append(&rand_vec(d, &mut rng), &rand_vec(d, &mut rng));
                assert!(cache.len() <= budget, "budget exceeded");
            }
            let q = rand_vec(d, &mut rng);
            (cache.len(), cache.evicted(), cache.attention(&q))
        };
        let (len, evicted, out) = run();
        assert_eq!(len, budget);
        assert_eq!(evicted, total - budget);
        let (len2, evicted2, out2) = run();
        assert_eq!((len, evicted), (len2, evicted2));
        assert_eq!(
            out.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            out2.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
}
