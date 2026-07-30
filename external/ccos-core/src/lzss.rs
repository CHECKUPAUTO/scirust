//! A small, dependency-free **lossless** byte codec (LZSS) for the COLD spill
//! store. Code and prose spill blobs are highly repetitive (`pub fn`, indentation,
//! identifiers), so a sliding-window LZ shrinks them on disk without the lossy
//! compaction of slice 4.
//!
//! Format: a 1-byte header — `0` = the payload is the original bytes verbatim
//! (used whenever compression wouldn't help, so the codec **never inflates**), `1` =
//! LZSS. The LZSS stream is the classic flag-byte scheme: one flag byte precedes up
//! to 8 tokens; bit *i* (LSB first) is `0` for a literal byte or `1` for a 2-byte
//! back-reference packing a 12-bit offset (1..4096) and a 4-bit length (3..18).
//!
//! **Safety net:** the spill store keys and verifies blobs by the SHA-256 of the
//! *original* content, re-checked on read. So even a latent bug here can only ever
//! produce a hash mismatch — a recoverable cold-miss — never silent corruption.
//! That is what makes a hand-rolled codec acceptable on the lossless path; the
//! round-trip property test below is the primary guard.

const WINDOW: usize = 4096; // 12-bit offset
const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 18; // 4-bit length + MIN_MATCH
const MAX_CHAIN: usize = 64; // hash-chain search-depth cap (bounds compress time)
const HASH_SIZE: usize = 1 << 13;

/// Compress `data`, never inflating: returns `[0] ++ data` if LZSS wouldn't be
/// smaller, else `[1] ++ lzss(data)`. Deterministic (so content-addressing still
/// deduplicates identical blobs to one file).
pub fn compress(data: &[u8]) -> Vec<u8> {
    let packed = lzss_compress(data);
    let mut out = Vec::with_capacity(packed.len().min(data.len()) + 1);
    if packed.len() < data.len() {
        out.push(1);
        out.extend_from_slice(&packed);
    } else {
        out.push(0);
        out.extend_from_slice(data);
    }
    out
}

/// Inverse of [`compress`]. `None` on a malformed blob (unknown header, truncated
/// or out-of-range back-reference) — surfaced by the caller as a cold-miss.
pub fn decompress(blob: &[u8]) -> Option<Vec<u8>> {
    match blob.split_first() {
        Some((0, rest)) => Some(rest.to_vec()),
        Some((1, rest)) => lzss_decompress(rest),
        _ => None,
    }
}

fn hash3(data: &[u8], pos: usize) -> usize {
    let a = data[pos] as usize;
    let b = data[pos + 1] as usize;
    let c = data[pos + 2] as usize;
    ((a << 10) ^ (b << 5) ^ c) & (HASH_SIZE - 1)
}

fn lzss_compress(data: &[u8]) -> Vec<u8> {
    let n = data.len();
    let mut out = Vec::new();
    // Hash-chain index: `head[h]` is the most recent position whose 3-byte prefix
    // hashes to `h`; `prev[p]` chains to the prior such position. Bounds the match
    // search to MAX_CHAIN candidates per position instead of scanning the window.
    let mut head = vec![-1i32; HASH_SIZE];
    let mut prev = vec![-1i32; n];

    let insert = |head: &mut [i32], prev: &mut [i32], pos: usize| {
        if pos + MIN_MATCH <= n {
            let h = hash3(data, pos);
            prev[pos] = head[h];
            head[h] = pos as i32;
        }
    };

    let mut pos = 0;
    while pos < n {
        let flag_idx = out.len();
        out.push(0u8);
        let mut flag = 0u8;
        for bit in 0..8 {
            if pos >= n {
                break;
            }
            let max_len = (n - pos).min(MAX_MATCH);
            let mut best_len = 0usize;
            let mut best_off = 0usize;
            if max_len >= MIN_MATCH {
                let win_start = pos.saturating_sub(WINDOW);
                let mut cand = head[hash3(data, pos)];
                let mut chain = 0;
                while cand >= 0 && (cand as usize) >= win_start && chain < MAX_CHAIN {
                    let c = cand as usize;
                    let mut l = 0;
                    while l < max_len && data[c + l] == data[pos + l] {
                        l += 1;
                    }
                    if l > best_len {
                        best_len = l;
                        best_off = pos - c;
                        if l == max_len {
                            break;
                        }
                    }
                    cand = prev[c];
                    chain += 1;
                }
            }
            insert(&mut head, &mut prev, pos);
            if best_len >= MIN_MATCH {
                flag |= 1 << bit;
                let code = (((best_off - 1) as u16) << 4) | ((best_len - MIN_MATCH) as u16);
                out.push((code >> 8) as u8);
                out.push((code & 0xFF) as u8);
                for p in (pos + 1)..(pos + best_len) {
                    insert(&mut head, &mut prev, p);
                }
                pos += best_len;
            } else {
                out.push(data[pos]);
                pos += 1;
            }
        }
        out[flag_idx] = flag;
    }
    out
}

fn lzss_decompress(blob: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < blob.len() {
        let flag = blob[i];
        i += 1;
        for bit in 0..8 {
            if i >= blob.len() {
                break; // a short final block: remaining flag bits have no tokens
            }
            if flag & (1 << bit) != 0 {
                if i + 1 >= blob.len() {
                    return None; // truncated back-reference
                }
                let code = ((blob[i] as u16) << 8) | (blob[i + 1] as u16);
                i += 2;
                let off = ((code >> 4) + 1) as usize;
                let len = ((code & 0xF) as usize) + MIN_MATCH;
                if off > out.len() {
                    return None; // back-reference before the start of output
                }
                let start = out.len() - off;
                for k in 0..len {
                    out.push(out[start + k]); // byte-by-byte: handles off < len overlap
                }
            } else {
                out.push(blob[i]);
                i += 1;
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn round_trip(data: &[u8]) {
        let blob = compress(data);
        assert_eq!(
            decompress(&blob).as_deref(),
            Some(data),
            "round-trip for {data:?}"
        );
        // Never inflates beyond the 1-byte header.
        assert!(
            blob.len() <= data.len() + 1,
            "inflated {} → {}",
            data.len(),
            blob.len()
        );
    }

    #[test]
    fn round_trips_edge_cases() {
        round_trip(b"");
        round_trip(b"a");
        round_trip(b"ab");
        round_trip(b"abc");
        round_trip(&[0u8; 5000]); // long run → overlap matches
        round_trip(b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"); // overlap (off < len)
        round_trip(b"pub fn foo() {}\npub fn bar() {}\npub fn baz() {}\n");
    }

    #[test]
    fn compresses_repetitive_code() {
        let src = "pub fn function_x() -> u32 { 0 }\n".repeat(64);
        let blob = compress(src.as_bytes());
        assert_eq!(decompress(&blob).as_deref(), Some(src.as_bytes()));
        assert!(
            blob.len() * 2 < src.len(),
            "expected >2x on repetitive code, got {} → {}",
            src.len(),
            blob.len()
        );
    }

    #[test]
    fn rejects_malformed() {
        assert_eq!(decompress(&[]), None); // no header
        assert_eq!(decompress(&[2, 0, 0]), None); // unknown header
                                                  // Flag bit 0 = back-reference, but only 1 of its 2 bytes follows (truncated).
        assert_eq!(decompress(&[1, 0b0000_0001, 0x00]), None);
        // Back-reference whose offset points before the start of output.
        assert_eq!(decompress(&[1, 0b0000_0001, 0x00, 0x00]), None);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(400))]

        /// The core lossless guarantee: decompress(compress(x)) == x for ANY bytes.
        #[test]
        fn decompress_inverts_compress(data in prop::collection::vec(any::<u8>(), 0..2048)) {
            let blob = compress(&data);
            let got = decompress(&blob);
            prop_assert_eq!(got.as_deref(), Some(data.as_slice()));
        }

        /// Biased toward small alphabets (where matches abound) to stress the
        /// hash-chain / overlap paths harder than uniform random would.
        #[test]
        fn decompress_inverts_compress_low_entropy(
            data in prop::collection::vec(0u8..4u8, 0..3000)
        ) {
            let blob = compress(&data);
            let got = decompress(&blob);
            prop_assert_eq!(got.as_deref(), Some(data.as_slice()));
            prop_assert!(blob.len() <= data.len() + 1);
        }
    }
}
