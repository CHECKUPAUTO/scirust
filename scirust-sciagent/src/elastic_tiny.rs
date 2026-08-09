//! Small-piece BPE kernel for the ElasticTokenizer.
//!
//! `TinyScanBpe` uses a fixed stack work buffer and the same global rank
//! priority rule as `CanonicalBpeOracle`. It never chunks an oversized piece:
//! callers receive `None` and must route the complete piece to another kernel.

use std::collections::BTreeMap;

use crate::elastic_id::{PackedRule, PriorityKey};
use crate::elastic_rule_table::FlatPackedRuleTable;
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

#[derive(Clone, Debug)]
enum RuleTable {
    Compact(FlatPackedRuleTable),
    Wide(BTreeMap<(TokenId, TokenId), TinyRule>),
}

impl RuleTable {
    fn from_ordered_merges(
        merges: &[(TokenId, TokenId, TokenId)],
    ) -> Result<Self, DuplicateMergeRule> {
        if let Some(ranked) = FlatPackedRuleTable::try_from_ordered_merges(merges)?
        {
            return Ok(Self::Compact(ranked));
        }

        let mut ranked = BTreeMap::new();
        for (rank, &(left, right, output)) in merges.iter().enumerate()
        {
            if ranked
                .insert((left, right), TinyRule { output, rank })
                .is_some()
            {
                return Err(DuplicateMergeRule { left, right });
            }
        }
        Ok(Self::Wide(ranked))
    }

    fn is_compact(&self) -> bool {
        matches!(self, Self::Compact(_))
    }

    fn get_compact(&self, left: u32, right: u32) -> Option<PackedRule> {
        match self
        {
            Self::Compact(rules) => rules.get(left, right),
            Self::Wide(_) => None,
        }
    }

    fn get(&self, left: TokenId, right: TokenId) -> Option<TinyRule> {
        match self
        {
            Self::Compact(rules) =>
            {
                let left = u32::try_from(left).ok()?;
                let right = u32::try_from(right).ok()?;
                let rule = rules.get(left, right)?;
                Some(TinyRule {
                    output: usize::try_from(rule.output()).ok()?,
                    rank: usize::try_from(rule.rank()).ok()?,
                })
            },
            Self::Wide(rules) => rules.get(&(left, right)).copied(),
        }
    }
}

/// Rank-priority BPE kernel specialized for small pieces.
#[derive(Clone, Debug)]
pub struct TinyScanBpe {
    merges: RuleTable,
}

impl TinyScanBpe {
    pub fn from_ordered_merges(
        merges: &[(TokenId, TokenId, TokenId)],
    ) -> Result<Self, DuplicateMergeRule> {
        Ok(Self {
            merges: RuleTable::from_ordered_merges(merges)?,
        })
    }

    /// Encodes a complete piece when it fits the tiny work buffer.
    ///
    /// Normal tokenizer IDs use a 512-byte `[u32; 128]` stack buffer on every
    /// target. Inputs outside the compact ID domain execute the historical wide
    /// path for the complete piece instead of being truncated or split.
    pub fn try_encode_ids(&self, input: &[TokenId]) -> Option<Vec<TokenId>> {
        if input.len() > TINY_SCAN_CAPACITY
        {
            return None;
        }
        if self.merges.is_compact() && input.iter().all(|&token| u32::try_from(token).is_ok())
        {
            Some(self.encode_compact(input))
        }
        else
        {
            Some(self.encode_wide(input))
        }
    }

    fn encode_compact(&self, input: &[TokenId]) -> Vec<TokenId> {
        let mut work = [0u32; TINY_SCAN_CAPACITY];
        for (slot, &token) in work.iter_mut().zip(input)
        {
            *slot = u32::try_from(token).expect("compact input preflight checked token ids");
        }
        let mut len = input.len();

        while len >= 2
        {
            let mut best: Option<(PriorityKey, u32)> = None;

            for position in 0..len - 1
            {
                let Some(rule) = self.merges.get_compact(work[position], work[position + 1])
                else
                {
                    continue;
                };
                let position = u32::try_from(position).expect("TinyScan position fits u32");
                let candidate = (PriorityKey::new(rule.rank(), position), rule.output());
                if best.is_none_or(|current| candidate.0 < current.0)
                {
                    best = Some(candidate);
                }
            }

            let Some((priority, output)) = best
            else
            {
                break;
            };
            let position =
                usize::try_from(priority.left_index()).expect("TinyScan position fits usize");
            work[position] = output;
            work.copy_within(position + 2..len, position + 1);
            len -= 1;
        }

        work[..len]
            .iter()
            .copied()
            .map(|token| usize::try_from(token).expect("u32 token id fits usize"))
            .collect()
    }

    fn encode_wide(&self, input: &[TokenId]) -> Vec<TokenId> {
        let mut work = [0; TINY_SCAN_CAPACITY];
        work[..input.len()].copy_from_slice(input);
        let mut len = input.len();

        while len >= 2
        {
            let mut best: Option<(usize, usize, TokenId)> = None;

            for position in 0..len - 1
            {
                let Some(rule) = self.merges.get(work[position], work[position + 1])
                else
                {
                    continue;
                };
                let candidate = (rule.rank, position, rule.output);
                if best.is_none_or(|current| (candidate.0, candidate.1) < (current.0, current.1))
                {
                    best = Some(candidate);
                }
            }

            let Some((_, position, output)) = best
            else
            {
                break;
            };

            work[position] = output;
            work.copy_within(position + 2..len, position + 1);
            len -= 1;
        }

        work[..len].to_vec()
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
    fn compact_stack_buffer_is_half_width_on_64_bit_hosts() {
        assert_eq!(std::mem::size_of::<[u32; TINY_SCAN_CAPACITY]>(), 512);
        if usize::BITS == 64
        {
            assert_eq!(std::mem::size_of::<[TokenId; TINY_SCAN_CAPACITY]>(), 1024);
        }
    }

    #[test]
    fn normal_rule_tables_use_flat_compact_storage() {
        let tiny = TinyScanBpe::from_ordered_merges(&[(1, 2, 3), (3, 4, 5)]).unwrap();
        assert!(matches!(tiny.merges, RuleTable::Compact(_)));
    }

    #[test]
    fn tiny_scan_matches_rank_priority_conflict() {
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

        for len in 0..=7
        {
            let cases = 3usize.pow(len as u32);
            for mut encoded in 0..cases
            {
                let mut input = vec![1; len];
                for token in &mut input
                {
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
    fn wide_id_fallback_preserves_semantics() {
        if usize::BITS > 32
        {
            let wide = usize::try_from(u64::from(u32::MAX) + 1).unwrap();
            let tiny = TinyScanBpe::from_ordered_merges(&[(wide, 1, 2)]).unwrap();
            assert!(matches!(tiny.merges, RuleTable::Wide(_)));
            assert_eq!(tiny.try_encode_ids(&[wide, 1]).unwrap(), vec![2]);
        }
    }

    #[test]
    fn duplicate_rules_are_rejected_consistently() {
        let err = TinyScanBpe::from_ordered_merges(&[(1, 2, 3), (1, 2, 4)]).unwrap_err();
        assert_eq!(err, DuplicateMergeRule { left: 1, right: 2 });
    }
}
