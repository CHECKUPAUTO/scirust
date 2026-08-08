//! Small-piece BPE kernel for the ElasticTokenizer.
//!
//! `TinyScanBpe` uses a fixed stack work buffer and the same global rank
//! priority rule as `CanonicalBpeOracle`. It never chunks an oversized piece:
//! callers receive `None` and must route the complete piece to another kernel.

use std::collections::BTreeMap;

use crate::elastic_tokenizer::{DuplicateMergeRule, TokenId};

/// Maximum number of input ids handled by the stack-only tiny work buffer.
///
/// This is an implementation capacity, not a semantic BPE boundary. The
/// auto-calibrator may choose a much smaller routing threshold.
pub const TINY_SCAN_CAPACITY: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TinyRule {
    output: TokenId,
    rank: usize,
}

/// Rank-priority BPE kernel specialized for small pieces.
#[derive(Clone, Debug)]
pub struct TinyScanBpe {
    merges: BTreeMap<(TokenId, TokenId), TinyRule>,
}

impl TinyScanBpe {
    pub fn from_ordered_merges(
        merges: &[(TokenId, TokenId, TokenId)],
    ) -> Result<Self, DuplicateMergeRule> {
        let mut ranked = BTreeMap::new();
        for (rank, &(left, right, output)) in merges.iter().enumerate() {
            if ranked
                .insert((left, right), TinyRule { output, rank })
                .is_some()
            {
                return Err(DuplicateMergeRule { left, right });
            }
        }
        Ok(Self { merges: ranked })
    }

    /// Encodes a complete piece when it fits the tiny work buffer.
    ///
    /// The work set itself is stack-resident. The returned `Vec` is the only
    /// per-call allocation performed by this API; a later scratch/output API
    /// can remove that final allocation without changing semantics.
    pub fn try_encode_ids(&self, input: &[TokenId]) -> Option<Vec<TokenId>> {
        if input.len() > TINY_SCAN_CAPACITY {
            return None;
        }

        let mut work = [0; TINY_SCAN_CAPACITY];
        work[..input.len()].copy_from_slice(input);
        let mut len = input.len();

        while len >= 2 {
            let mut best: Option<(usize, usize, TokenId)> = None;

            for position in 0..len - 1 {
                let pair = (work[position], work[position + 1]);
                let Some(rule) = self.merges.get(&pair) else {
                    continue;
                };
                let candidate = (rule.rank, position, rule.output);
                if best
                    .map(|current| (candidate.0, candidate.1) < (current.0, current.1))
                    .unwrap_or(true)
                {
                    best = Some(candidate);
                }
            }

            let Some((_, position, output)) = best else {
                break;
            };

            work[position] = output;
            work.copy_within(position + 2..len, position + 1);
            len -= 1;
        }

        Some(work[..len].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elastic_tokenizer::CanonicalBpeOracle;

    fn assert_parity(merges: &[(TokenId, TokenId, TokenId)], input: &[TokenId]) {
        let reference = CanonicalBpeOracle::from_ordered_merges(merges).unwrap();
        let tiny = TinyScanBpe::from_ordered_merges(merges).unwrap();
        assert_eq!(
            tiny.try_encode_ids(input).unwrap(),
            reference.encode_ids(input)
        );
    }

    #[test]
    fn tiny_scan_matches_rank_priority_conflict() {
        // `2+3` outranks `1+2`; the kernel must not greedily merge from the left.
        let merges = [(2, 3, 10), (1, 2, 11)];
        assert_parity(&merges, &[1, 2, 3]);
    }

    #[test]
    fn tiny_scan_matches_recursive_merges() {
        let merges = [(1, 2, 10), (10, 3, 11), (11, 4, 12), (2, 3, 13)];
        assert_parity(&merges, &[1, 2, 3, 4]);
    }

    #[test]
    fn tiny_scan_exhaustive_small_alphabet_parity() {
        // Exhaust every sequence over {1,2,3} up to length 7. The table has
        // overlaps and recursive merges, so this catches rank and invalidated
        // adjacency mistakes without relying on RNG behavior.
        let merges = [
            (2, 3, 10),
            (1, 2, 11),
            (1, 10, 12),
            (11, 3, 13),
            (3, 3, 14),
            (14, 1, 15),
        ];
        let reference = CanonicalBpeOracle::from_ordered_merges(&merges).unwrap();
        let tiny = TinyScanBpe::from_ordered_merges(&merges).unwrap();

        for len in 0..=7 {
            let cases = 3usize.pow(len as u32);
            for mut encoded in 0..cases {
                let mut input = vec![1; len];
                for token in &mut input {
                    *token = encoded % 3 + 1;
                    encoded /= 3;
                }
                assert_eq!(
                    tiny.try_encode_ids(&input).unwrap(),
                    reference.encode_ids(&input),
                    "parity failed for input {input:?}"
                );
            }
        }
    }

    #[test]
    fn oversized_piece_is_rejected_not_chunked() {
        let tiny = TinyScanBpe::from_ordered_merges(&[(1, 1, 2)]).unwrap();
        let input = vec![1; TINY_SCAN_CAPACITY + 1];
        assert_eq!(tiny.try_encode_ids(&input), None);
    }

    #[test]
    fn duplicate_rules_are_rejected_consistently() {
        let err = TinyScanBpe::from_ordered_merges(&[(1, 2, 3), (1, 2, 4)]).unwrap_err();
        assert_eq!(err, DuplicateMergeRule { left: 1, right: 2 });
    }
}
