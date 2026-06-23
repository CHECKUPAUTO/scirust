//! Huffman coding — entropy-optimal prefix-free code for symbol frequencies.
//!
//! Reversible: [`Encoder::encode`] produces a bit string, [`Decoder::decode`]
//! reconstructs the original symbol sequence exactly. Combined with the
//! trie/MinHash layers, this is the last-mile bit-packing for text chunks that
//! have already been deduplicated and structurally compacted.
//!
//! Deterministic: tie-breaks on the symbol byte value so two equal-frequency
//! histograms produce identical codebooks.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BinaryHeap};

/// A Huffman codebook + canonical encoding table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuffmanCodebook {
    /// Symbol (byte) → (code length, canonical code value).
    pub codes: BTreeMap<u8, (u32, u64)>,
    /// Max code length (bits), for decoder sizing.
    pub max_len: u32,
}

/// Reversible encoder holding a codebook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Encoder {
    pub codebook: HuffmanCodebook,
}

/// Reversible decoder holding the same codebook, indexed for fast lookup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decoder {
    pub codebook: HuffmanCodebook,
    /// (code length, code value) → symbol, for decode lookup.
    #[serde(skip)]
    pub lookup: BTreeMap<(u32, u64), u8>,
}

#[derive(Debug, Eq, PartialEq)]
struct HeapNode {
    freq: u64,
    seq: u64,
    symbol: Option<u8>,
    left: Option<Box<HeapNode>>,
    right: Option<Box<HeapNode>>,
}

impl Ord for HeapNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap by (freq, seq): deterministic tie-break on insertion order.
        other
            .freq
            .cmp(&self.freq)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

impl PartialOrd for HeapNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Build a Huffman codebook from a symbol frequency table.
///
/// `freqs` maps each byte to its occurrence count. Bytes with zero frequency
/// are omitted from the codebook. When `freqs` has a single symbol, it is
/// assigned a 1-bit code (the tree is a single leaf with a synthetic sibling).
pub fn build_codebook(freqs: &BTreeMap<u8, u64>) -> HuffmanCodebook {
    if freqs.is_empty()
    {
        return HuffmanCodebook {
            codes: BTreeMap::new(),
            max_len: 0,
        };
    }
    // Build the tree.
    let mut heap = BinaryHeap::new();
    let mut seq = 0u64;
    for (&sym, &freq) in freqs
    {
        heap.push(HeapNode {
            freq,
            seq,
            symbol: Some(sym),
            left: None,
            right: None,
        });
        seq += 1;
    }
    // Single-symbol case: assign a 1-bit code.
    if freqs.len() == 1
    {
        let sym = *freqs.keys().next().unwrap();
        let mut codes = BTreeMap::new();
        codes.insert(sym, (1u32, 0u64));
        return HuffmanCodebook { codes, max_len: 1 };
    }
    while heap.len() > 1
    {
        let a = heap.pop().unwrap();
        let b = heap.pop().unwrap();
        heap.push(HeapNode {
            freq: a.freq + b.freq,
            seq,
            symbol: None,
            left: Some(Box::new(a)),
            right: Some(Box::new(b)),
        });
        seq += 1;
    }
    let root = heap.pop().unwrap();
    // Walk the tree to get (length, raw code). Then canonicalize so codes are
    // assigned by (length, symbol) order — this makes the codebook independent
    // of tie-break luck and ensures decoder unambiguity.
    let mut lengths: BTreeMap<u8, u32> = BTreeMap::new();
    walk(&root, 0, &mut lengths);
    canonicalize(&lengths)
}

fn walk(node: &HeapNode, depth: u32, out: &mut BTreeMap<u8, u32>) {
    if let Some(sym) = node.symbol
    {
        out.insert(sym, depth.max(1));
        return;
    }
    if let Some(l) = &node.left
    {
        walk(l, depth + 1, out);
    }
    if let Some(r) = &node.right
    {
        walk(r, depth + 1, out);
    }
}

/// Convert a length map into canonical Huffman codes: sort by (length, symbol)
/// and assign sequential binary values.
fn canonicalize(lengths: &BTreeMap<u8, u32>) -> HuffmanCodebook {
    let mut by_len: BTreeMap<u32, Vec<u8>> = BTreeMap::new();
    for (&sym, &len) in lengths
    {
        by_len.entry(len).or_default().push(sym);
    }
    for v in by_len.values_mut()
    {
        v.sort();
    }
    let mut codes: BTreeMap<u8, (u32, u64)> = BTreeMap::new();
    let mut code: u64 = 0;
    let mut prev_len: u32 = 0;
    let mut max_len = 0u32;
    for (&len, symbols) in by_len.iter()
    {
        if prev_len != 0
        {
            code <<= len - prev_len;
        }
        for &sym in symbols
        {
            codes.insert(sym, (len, code));
            code += 1;
            if len > max_len
            {
                max_len = len;
            }
        }
        prev_len = len;
    }
    HuffmanCodebook { codes, max_len }
}

impl Encoder {
    /// Build an encoder from a frequency table (see [`build_codebook`]).
    pub fn from_freqs(freqs: &BTreeMap<u8, u64>) -> Self {
        Self {
            codebook: build_codebook(freqs),
        }
    }

    /// Build an encoder by counting bytes in `data`.
    pub fn from_data(data: &[u8]) -> Self {
        let mut freqs = BTreeMap::new();
        for &b in data
        {
            *freqs.entry(b).or_insert(0u64) += 1;
        }
        Self::from_freqs(&freqs)
    }

    /// Encode `data` into a packed bit vector (`u64` words, MSB-first within
    /// each code). Returns the bits plus the bit count (the last word may be
    /// padded with zeros).
    pub fn encode(&self, data: &[u8]) -> (Vec<u64>, usize) {
        let mut bits: Vec<u64> = vec![0u64];
        let mut used: usize = 0; // bits written in the current word
        let mut word_idx: usize = 0;
        for &b in data
        {
            let (len, code) = match self.codebook.codes.get(&b)
            {
                Some(&c) => c,
                None => continue, // symbol not in codebook (shouldn't happen for self-built)
            };
            for i in (0..len).rev()
            {
                let bit = (code >> i) & 1;
                bits[word_idx] |= bit << (63 - used);
                used += 1;
                if used == 64
                {
                    bits.push(0u64);
                    word_idx += 1;
                    used = 0;
                }
            }
        }
        let total_bits = word_idx * 64 + used;
        (bits, total_bits)
    }

    /// Estimated encoded size in bytes (ceil of total_bits / 8).
    pub fn encoded_bytes(&self, data: &[u8]) -> usize {
        let mut total_bits = 0usize;
        for &b in data
        {
            if let Some(&(len, _)) = self.codebook.codes.get(&b)
            {
                total_bits += len as usize;
            }
        }
        total_bits.div_ceil(8)
    }
}

impl Decoder {
    /// Build a decoder from a codebook (typically deserialized).
    pub fn from_codebook(codebook: HuffmanCodebook) -> Self {
        let mut lookup = BTreeMap::new();
        for (&sym, &(len, code)) in &codebook.codes
        {
            lookup.insert((len, code), sym);
        }
        Self { codebook, lookup }
    }

    /// Decode `bits` (with `total_bits` valid bits) back into the original
    /// byte sequence. Errors with `None` on a non-decodable bit sequence
    /// (truncated or unknown code).
    pub fn decode(&self, bits: &[u64], total_bits: usize) -> Option<Vec<u8>> {
        let mut out = Vec::new();
        let mut code: u64 = 0;
        let mut len: u32 = 0;
        for i in 0..total_bits
        {
            let word = bits[i / 64];
            let bit = (word >> (63 - (i % 64))) & 1;
            code = (code << 1) | bit;
            len += 1;
            if len > self.codebook.max_len
            {
                return None; // no code this long → corrupt
            }
            if let Some(&sym) = self.lookup.get(&(len, code))
            {
                out.push(sym);
                code = 0;
                len = 0;
            }
        }
        if len != 0
        {
            return None; // trailing bits don't form a complete code
        }
        Some(out)
    }
}

/// Round-trip helper: encode then immediately decode `data`.
pub fn round_trip(data: &[u8]) -> Vec<u8> {
    let enc = Encoder::from_data(data);
    let (bits, n) = enc.encode(data);
    let dec = Decoder::from_codebook(enc.codebook);
    dec.decode(&bits, n).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_random_text() {
        let text = b"the quick brown fox jumps over the lazy dog repeated many times";
        let dec = round_trip(text);
        assert_eq!(dec, text);
    }

    #[test]
    fn round_trip_single_symbol() {
        let dec = round_trip(b"aaaaaaa");
        assert_eq!(dec, b"aaaaaaa");
    }

    #[test]
    fn round_trip_all_byte_values() {
        let data: Vec<u8> = (0u8..=255).collect();
        let dec = round_trip(&data);
        assert_eq!(dec, data);
    }

    #[test]
    fn skew_compresses_better_than_uniform() {
        // Highly skewed: 'a' dominates → very short code for 'a'.
        let mut skewed = vec![b'a'; 1000];
        skewed.push(b'z');
        let enc = Encoder::from_data(&skewed);
        let size = enc.encoded_bytes(&skewed);
        // 1000 'a's at 1 bit + 1 'z' at ~10 bits ≈ ~127 bytes vs 1001 raw.
        assert!(size < 200, "skewed should compress well: {size} bytes");

        // Uniform: no compression gain.
        let uniform: Vec<u8> = (0u8..=255).cycle().take(1000).collect();
        let enc2 = Encoder::from_data(&uniform);
        let size2 = enc2.encoded_bytes(&uniform);
        // All 8-bit codes → ~1000 bytes (no gain, slight overhead).
        assert!(size2 >= 990, "uniform shouldn't compress: {size2}");
    }

    #[test]
    fn determinism_same_histogram_same_codebook() {
        let mut freqs = BTreeMap::new();
        for &(s, f) in &[(b'a', 5u64), (b'b', 3), (b'c', 2), (b'd', 2)]
        {
            freqs.insert(s, f);
        }
        let a = build_codebook(&freqs);
        let b = build_codebook(&freqs);
        assert_eq!(a.codes, b.codes, "same histogram → same canonical codes");
    }

    #[test]
    fn empty_input_round_trips() {
        let dec = round_trip(b"");
        assert!(dec.is_empty());
    }

    #[test]
    fn truncated_decode_returns_none() {
        let enc = Encoder::from_data(b"hello world");
        let (bits, n) = enc.encode(b"hello world");
        // Drop the last bit → decode must fail.
        let dec = Decoder::from_codebook(enc.codebook);
        assert!(dec.decode(&bits, n - 1).is_none());
    }
}
