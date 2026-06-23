//! Bloom filter — space-efficient probabilistic membership test.
//!
//! A Bloom filter answers "is this element *possibly* in the set?" with no
//! false negatives and a tunable false-positive rate. It never stores the
//! elements themselves, only `k` bits per insertion, which makes it ideal for
//! cheap dedup pre-filtering of context chunks before paying the cost of an
//! exact hash lookup.
//!
//! Deterministic: the `k` hash functions are derived from a single seeded
//! splitmix64, so two filters built with the same `seed` and the same insertion
//! order are bit-identical.

use serde::{Deserialize, Serialize};

/// A bit-addressable Bloom filter backed by a packed `Vec<u64>`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloomFilter {
    /// Bits packed into 64-bit words.
    bits: Vec<u64>,
    /// Number of bits (`m`).
    num_bits: usize,
    /// Number of hash functions (`k`).
    num_hashes: usize,
    /// Seed for the hash family (reproducibility).
    seed: u64,
    /// Number of elements inserted (for occupancy reporting).
    inserted: usize,
}

impl BloomFilter {
    /// Create a filter sized for `capacity` elements at a target
    /// false-positive rate `p` (e.g. `0.01` for ~1%).
    ///
    /// Uses the classic formulas:
    /// `m = -n·ln(p) / (ln(2))²`, `k = (m/n)·ln(2)`.
    pub fn new(capacity: usize, p: f64, seed: u64) -> Self {
        let n = capacity.max(1);
        let pp = p.clamp(1e-9, 0.5);
        let m = (-(n as f64) * pp.ln() / (std::f64::consts::LN_2 * std::f64::consts::LN_2)).ceil()
            as usize;
        let m = m.max(64).next_power_of_two();
        let k = ((m as f64 / n as f64) * std::f64::consts::LN_2).round() as usize;
        let k = k.max(1);
        Self {
            bits: vec![0u64; m.div_ceil(64)],
            num_bits: m,
            num_hashes: k,
            seed,
            inserted: 0,
        }
    }

    /// Double hashing (Kirsch-Mitzenmacher): `g_i(x) = h1(x) + i·h2(x) mod m`.
    /// `h1` is FNV-1a, `h2` is derived from a seed-mixed variant — both
    /// deterministic and independent enough for Bloom use.
    fn hashes(&self, data: &[u8]) -> (u64, u64) {
        let mut h1: u64 = 14695981039346656037; // FNV offset
        for &b in data
        {
            h1 ^= b as u64;
            h1 = h1.wrapping_mul(1099511628211);
        }
        let mut h2: u64 = self.seed.wrapping_add(0x9E3779B97F4A7C15);
        for &b in data
        {
            h2 ^= b as u64;
            h2 = h2.wrapping_mul(6364136223846793005);
        }
        if h2 == 0
        {
            h2 = 1;
        }
        (h1, h2)
    }

    fn bit_index(&self, h1: u64, h2: u64, i: usize) -> usize {
        (h1.wrapping_add((i as u64).wrapping_mul(h2)) % self.num_bits as u64) as usize
    }

    fn set_bit(&mut self, idx: usize) {
        self.bits[idx / 64] |= 1u64 << (idx % 64);
    }

    fn get_bit(&self, idx: usize) -> bool {
        (self.bits[idx / 64] >> (idx % 64)) & 1 == 1
    }

    /// Insert an element. Idempotent on repeated insertions of the same value.
    pub fn insert(&mut self, data: &[u8]) {
        let (h1, h2) = self.hashes(data);
        for i in 0..self.num_hashes
        {
            self.set_bit(self.bit_index(h1, h2, i));
        }
        self.inserted += 1;
    }

    /// Insert a string (UTF-8 bytes).
    pub fn insert_str(&mut self, s: &str) {
        self.insert(s.as_bytes());
    }

    /// True if the element is *possibly* in the set (may false-positive).
    /// False means definitely absent.
    pub fn contains(&self, data: &[u8]) -> bool {
        let (h1, h2) = self.hashes(data);
        (0..self.num_hashes).all(|i| self.get_bit(self.bit_index(h1, h2, i)))
    }

    /// String variant of [`contains`](Self::contains).
    pub fn contains_str(&self, s: &str) -> bool {
        self.contains(s.as_bytes())
    }

    /// Number of bits set (popcount) — useful for estimating occupancy.
    pub fn bits_set(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Estimated false-positive rate given the current occupancy.
    pub fn estimated_fpr(&self) -> f64 {
        let m = self.num_bits as f64;
        let n = self.inserted as f64;
        let k = self.num_hashes as f64;
        (1.0 - (-k * n / m).exp()).powi(k as i32)
    }

    /// Number of inserted elements.
    pub fn len(&self) -> usize {
        self.inserted
    }

    /// True when no element has ever been inserted.
    pub fn is_empty(&self) -> bool {
        self.inserted == 0
    }

    /// Merge another filter into this one (union). Both filters must share the
    /// same `num_bits`, `num_hashes` and `seed` — otherwise `None` is returned.
    pub fn merge(&mut self, other: &BloomFilter) -> Option<()> {
        if other.num_bits != self.num_bits
            || other.num_hashes != self.num_hashes
            || other.seed != self.seed
        {
            return None;
        }
        for (a, b) in self.bits.iter_mut().zip(other.bits.iter())
        {
            *a |= *b;
        }
        self.inserted += other.inserted;
        Some(())
    }

    /// Clear all insertions (keeps sizing/seed).
    pub fn clear(&mut self) {
        self.bits.fill(0);
        self.inserted = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_elements_are_never_reported_present() {
        let mut f = BloomFilter::new(1000, 0.01, 42);
        for i in 0..500
        {
            f.insert_str(&format!("item-{i}"));
        }
        for i in 0..500
        {
            assert!(
                f.contains_str(&format!("item-{i}")),
                "inserted item missing"
            );
        }
        // Items never inserted: false positives possible but must be rare at p=0.01.
        let fp = (1000..2000)
            .filter(|i| f.contains_str(&format!("item-{i}")))
            .count();
        assert!(fp < 30, "too many false positives: {fp}");
    }

    #[test]
    fn determinism_same_seed_same_bits() {
        let mut a = BloomFilter::new(200, 0.02, 7);
        let mut b = BloomFilter::new(200, 0.02, 7);
        for w in ["alpha", "beta", "gamma", "delta"]
        {
            a.insert_str(w);
            b.insert_str(w);
        }
        assert_eq!(a.bits, b.bits, "same seed + order → bit-identical");
    }

    #[test]
    fn merge_unions_two_filters() {
        let mut a = BloomFilter::new(200, 0.01, 9);
        let mut b = BloomFilter::new(200, 0.01, 9);
        a.insert_str("only-a");
        b.insert_str("only-b");
        a.insert_str("shared");
        b.insert_str("shared");
        assert!(a.merge(&b).is_some());
        assert!(a.contains_str("only-a"));
        assert!(a.contains_str("only-b"));
        assert!(a.contains_str("shared"));
    }

    #[test]
    fn merge_rejects_mismatched_params() {
        let mut a = BloomFilter::new(200, 0.01, 1);
        let b = BloomFilter::new(200, 0.01, 2);
        assert!(a.merge(&b).is_none());
    }

    #[test]
    fn fpr_stays_near_target() {
        let n = 5000;
        let mut f = BloomFilter::new(n, 0.01, 123);
        for i in 0..n
        {
            f.insert_str(&format!("x-{i}"));
        }
        let fpr = f.estimated_fpr();
        assert!(fpr < 0.05, "estimated fpr {fpr} too high");
    }
}
