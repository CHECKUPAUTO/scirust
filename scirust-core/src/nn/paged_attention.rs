//! **PagedAttention** (Kwon et al., *vLLM*, SOSP 2023, arXiv:2309.06180).
//!
//! Autoregressive decoding caches the keys and values of every past token. Storing
//! that KV-cache **contiguously** per sequence wastes memory: each sequence must
//! reserve its maximum length up front, and the leftover is unusable by others
//! (internal + external fragmentation). PagedAttention borrows virtual-memory
//! **paging**: the cache is cut into fixed-size **blocks** scattered across a shared
//! pool, and a per-sequence **block table** maps logical positions to physical
//! blocks. Blocks fill on demand, so almost no memory is wasted and blocks can even
//! be shared across sequences.
//!
//! The cache is purely a *storage* layout — it must not change the numbers. This
//! module implements the paged KV store and an attention that indexes keys/values
//! **through the block table**, and the tests certify the defining guarantee: the
//! gathered cache and the paged-attention output are **bit-identical** to the
//! contiguous reference, even when the physical blocks are fragmented (non-
//! sequential). Pure, deterministic `f32` arithmetic; each KV vector has width `d`.

/// **Paged KV-cache** for one sequence: keys and values are stored in fixed-size
/// blocks drawn from an internal pool, addressed indirectly via a block table.
pub struct PagedKvCache {
    d: usize,
    block_size: usize,
    /// Physical key blocks (each `block_size · d` long); the value pool mirrors it.
    pool_k: Vec<Vec<f32>>,
    pool_v: Vec<Vec<f32>>,
    /// Logical block `i` of this sequence → physical index into the pools.
    block_table: Vec<usize>,
    len: usize,
}

impl PagedKvCache {
    /// Empty cache for KV vectors of width `d`, paged in blocks of `block_size`
    /// positions.
    pub fn new(d: usize, block_size: usize) -> Self {
        assert!(
            d > 0 && block_size > 0,
            "PagedKvCache: need d>0, block_size>0"
        );
        Self {
            d,
            block_size,
            pool_k: Vec::new(),
            pool_v: Vec::new(),
            block_table: Vec::new(),
            len: 0,
        }
    }

    /// Number of cached positions.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Physical blocks this sequence occupies (`⌈len / block_size⌉`).
    pub fn num_blocks(&self) -> usize {
        self.block_table.len()
    }

    /// Allocate a **decoy** physical block in the pool without mapping it into this
    /// sequence — simulating fragmentation, so the next real block lands at a
    /// non-sequential physical index. Used to prove the block table is honoured.
    pub fn reserve_decoy(&mut self) {
        self.pool_k.push(vec![f32::NAN; self.block_size * self.d]);
        self.pool_v.push(vec![f32::NAN; self.block_size * self.d]);
    }

    /// Append one position's key and value vectors (each length `d`), allocating a
    /// fresh physical block from the pool whenever the current logical block fills.
    pub fn append(&mut self, k: &[f32], v: &[f32]) {
        assert_eq!(k.len(), self.d, "append: key must be length d");
        assert_eq!(v.len(), self.d, "append: value must be length d");
        if self.len.is_multiple_of(self.block_size)
        {
            // New logical block → grab the next physical slot from the pool.
            let phys = self.pool_k.len();
            self.pool_k.push(vec![0.0f32; self.block_size * self.d]);
            self.pool_v.push(vec![0.0f32; self.block_size * self.d]);
            self.block_table.push(phys);
        }
        let within = self.len % self.block_size;
        let phys = self.block_table[self.len / self.block_size];
        let off = within * self.d;
        self.pool_k[phys][off..off + self.d].copy_from_slice(k);
        self.pool_v[phys][off..off + self.d].copy_from_slice(v);
        self.len += 1;
    }

    /// The physical slice of the key vector at logical position `t` (indexed through
    /// the block table).
    fn key_at(&self, t: usize) -> &[f32] {
        let phys = self.block_table[t / self.block_size];
        let off = (t % self.block_size) * self.d;
        &self.pool_k[phys][off..off + self.d]
    }

    /// The physical slice of the value vector at logical position `t`.
    fn value_at(&self, t: usize) -> &[f32] {
        let phys = self.block_table[t / self.block_size];
        let off = (t % self.block_size) * self.d;
        &self.pool_v[phys][off..off + self.d]
    }

    /// Reconstruct the contiguous `len×d` key matrix by walking the block table.
    pub fn gather_keys(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.len * self.d];
        for t in 0..self.len
        {
            out[t * self.d..(t + 1) * self.d].copy_from_slice(self.key_at(t));
        }
        out
    }

    /// Reconstruct the contiguous `len×d` value matrix.
    pub fn gather_values(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.len * self.d];
        for t in 0..self.len
        {
            out[t * self.d..(t + 1) * self.d].copy_from_slice(self.value_at(t));
        }
        out
    }

    /// Scaled-dot-product attention of a single query `q` (length `d`) over the
    /// cached keys/values, indexing them **through the block table**:
    /// `softmax(q·Kᵀ / √d) · V`. Returns the context vector (length `d`).
    pub fn attention(&self, q: &[f32]) -> Vec<f32> {
        assert_eq!(q.len(), self.d, "attention: query must be length d");
        attention_core(
            self.len,
            self.d,
            |t| self.key_at(t),
            |t| self.value_at(t),
            q,
        )
    }
}

/// Reference attention over a **contiguous** `n×d` key/value matrix — the layout
/// PagedAttention must reproduce exactly. Identical arithmetic order to
/// [`PagedKvCache::attention`].
pub fn contiguous_attention(
    keys: &[f32],
    values: &[f32],
    q: &[f32],
    d: usize,
    n: usize,
) -> Vec<f32> {
    assert_eq!(keys.len(), n * d);
    assert_eq!(values.len(), n * d);
    attention_core(
        n,
        d,
        |t| &keys[t * d..(t + 1) * d],
        |t| &values[t * d..(t + 1) * d],
        q,
    )
}

/// Shared attention kernel parameterised by how a position's key/value slice is
/// fetched, so the paged and contiguous paths run **bit-identical** arithmetic.
fn attention_core<'a>(
    n: usize,
    d: usize,
    key_at: impl Fn(usize) -> &'a [f32],
    value_at: impl Fn(usize) -> &'a [f32],
    q: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0f32; d];
    if n == 0
    {
        return out;
    }
    let scale = 1.0 / (d as f32).sqrt();
    // Scores qᵀkₜ / √d.
    let mut scores = vec![0.0f32; n];
    for (t, st) in scores.iter_mut().enumerate()
    {
        let k = key_at(t);
        let mut s = 0.0f32;
        for c in 0..d
        {
            s += q[c] * k[c];
        }
        *st = s * scale;
    }
    // Stable softmax.
    let mut mx = scores[0];
    for &s in &scores[1..]
    {
        if s > mx
        {
            mx = s;
        }
    }
    let mut denom = 0.0f32;
    for st in scores.iter_mut()
    {
        *st = (*st - mx).exp();
        denom += *st;
    }
    // Weighted sum of values.
    for (t, &score) in scores.iter().enumerate()
    {
        let p = score / denom;
        let v = value_at(t);
        for c in 0..d
        {
            out[c] += p * v[c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::PcgEngine;

    /// Build `n` random key/value vectors of width `d` from a seeded stream.
    fn random_kv(n: usize, d: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut rng = PcgEngine::new(seed);
        let k: Vec<f32> = (0..n * d).map(|_| rng.float_signed()).collect();
        let v: Vec<f32> = (0..n * d).map(|_| rng.float_signed()).collect();
        (k, v)
    }

    /// **Gather is lossless under fragmentation.** With decoy blocks interleaved so
    /// the physical layout is non-sequential, the gathered cache still reproduces
    /// the appended vectors **bit-for-bit**, and the block accounting is correct.
    #[test]
    fn gather_round_trips_through_fragmented_blocks() {
        let (n, d, bs) = (10usize, 3usize, 4usize);
        let (keys, values) = random_kv(n, d, 1);
        let mut cache = PagedKvCache::new(d, bs);
        for t in 0..n
        {
            // Fragment the pool before each new logical block.
            if t % bs == 0
            {
                cache.reserve_decoy();
            }
            cache.append(&keys[t * d..(t + 1) * d], &values[t * d..(t + 1) * d]);
        }
        assert_eq!(cache.len(), n);
        assert_eq!(cache.num_blocks(), n.div_ceil(bs)); // ⌈10/4⌉ = 3
        // Bit-identical round trip.
        let gk = cache.gather_keys();
        let gv = cache.gather_values();
        assert_eq!(
            gk.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            keys.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            gv.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            values.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }

    /// **The PagedAttention guarantee, tested.** Attention indexed through the
    /// (fragmented) block table is **bit-identical** to attention over the
    /// contiguous KV matrix — the paged layout is provably zero-cost numerically.
    #[test]
    fn paged_attention_equals_contiguous_bit_for_bit() {
        let (n, d, bs) = (13usize, 4usize, 4usize);
        let (keys, values) = random_kv(n, d, 9);
        let mut rng = PcgEngine::new(99);
        let q: Vec<f32> = (0..d).map(|_| rng.float_signed()).collect();

        let mut cache = PagedKvCache::new(d, bs);
        for t in 0..n
        {
            if t % bs == 0
            {
                cache.reserve_decoy(); // force non-contiguous physical blocks
            }
            cache.append(&keys[t * d..(t + 1) * d], &values[t * d..(t + 1) * d]);
        }
        // Physical blocks are genuinely non-sequential (decoys interleaved).
        assert!(cache.num_blocks() >= 4);

        let paged = cache.attention(&q);
        let reference = contiguous_attention(&keys, &values, &q, d, n);
        assert_eq!(
            paged.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            reference.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "paged attention diverged from contiguous"
        );
        // Determinism.
        assert_eq!(cache.attention(&q), paged);
    }

    /// Exact division into blocks (no partial last block) and the empty-cache edge
    /// case both behave.
    #[test]
    fn block_accounting_and_empty() {
        let (d, bs) = (2usize, 4usize);
        let cache = PagedKvCache::new(d, bs);
        assert!(cache.is_empty());
        assert_eq!(cache.attention(&[1.0, 2.0]), vec![0.0, 0.0]);

        let (keys, values) = random_kv(8, d, 3); // exactly 2 full blocks
        let mut cache = PagedKvCache::new(d, bs);
        for t in 0..8
        {
            cache.append(&keys[t * d..(t + 1) * d], &values[t * d..(t + 1) * d]);
        }
        assert_eq!(cache.num_blocks(), 2);
        assert_eq!(cache.gather_keys(), keys);
    }
}
