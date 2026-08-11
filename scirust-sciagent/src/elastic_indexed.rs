//! Indexed medium-piece BPE kernel for the ElasticTokenizer.
//!
//! `IndexedBpe` keeps the piece as an indexed linked list. Merge candidates
//! carry node generations so only adjacencies touched by a merge need to be
//! regenerated. Normal pieces use compact `u32` nodes and packed priorities;
//! a wide compatibility path preserves semantics for inputs outside that domain.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::elastic_id::{
    COMPACT_INDEX_INACTIVE, COMPACT_INDEX_NONE, PackedRule, PriorityKey, try_compact_index,
};
use crate::elastic_rule_table::AdaptivePackedRuleTable;
use crate::elastic_tokenizer::{DuplicateMergeRule, TokenId};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct IndexedRule {
    output: TokenId,
    rank: usize,
}

#[derive(Clone, Debug)]
enum RuleTable {
    Compact(Arc<AdaptivePackedRuleTable>),
    Wide(BTreeMap<(TokenId, TokenId), IndexedRule>),
}

impl RuleTable {
    fn from_ordered_merges(
        merges: &[(TokenId, TokenId, TokenId)],
    ) -> Result<Self, DuplicateMergeRule> {
        if let Some(rules) = AdaptivePackedRuleTable::try_from_ordered_merges(merges)?
        {
            return Ok(Self::Compact(Arc::new(rules)));
        }

        let mut ranked = BTreeMap::new();
        for (rank, &(left, right, output)) in merges.iter().enumerate()
        {
            if ranked
                .insert((left, right), IndexedRule { output, rank })
                .is_some()
            {
                return Err(DuplicateMergeRule { left, right });
            }
        }
        Ok(Self::Wide(ranked))
    }

    fn is_empty(&self) -> bool {
        match self
        {
            Self::Compact(rules) => rules.is_empty(),
            Self::Wide(rules) => rules.is_empty(),
        }
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

    fn get(&self, left: TokenId, right: TokenId) -> Option<IndexedRule> {
        match self
        {
            Self::Compact(rules) =>
            {
                let left = u32::try_from(left).ok()?;
                let right = u32::try_from(right).ok()?;
                let rule = rules.get(left, right)?;
                Some(IndexedRule {
                    output: usize::try_from(rule.output()).ok()?,
                    rank: usize::try_from(rule.rank()).ok()?,
                })
            },
            Self::Wide(rules) => rules.get(&(left, right)).copied(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CompactNode {
    token: u32,
    prev: u32,
    next: u32,
    generation: u32,
}

impl CompactNode {
    fn is_active(self) -> bool {
        self.next != COMPACT_INDEX_INACTIVE
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CompactCandidate {
    priority: PriorityKey,
    right: u32,
    left_generation: u32,
    right_generation: u32,
    output: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct WideNode {
    token: TokenId,
    prev: Option<usize>,
    next: Option<usize>,
    generation: u32,
    active: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct WideCandidate {
    rank: usize,
    left: usize,
    right: usize,
    left_generation: u32,
    right_generation: u32,
    output: TokenId,
}

/// Rank-priority BPE kernel with indexed adjacency maintenance.
#[derive(Clone, Debug)]
pub struct IndexedBpe {
    merges: RuleTable,
}

impl IndexedBpe {
    pub fn from_ordered_merges(
        merges: &[(TokenId, TokenId, TokenId)],
    ) -> Result<Self, DuplicateMergeRule> {
        Ok(Self {
            merges: RuleTable::from_ordered_merges(merges)?,
        })
    }

    pub(crate) fn from_shared_compact_rules(rules: Arc<AdaptivePackedRuleTable>) -> Self {
        Self {
            merges: RuleTable::Compact(rules),
        }
    }

    #[cfg(test)]
    pub(crate) fn compact_rule_table(&self) -> Option<&Arc<AdaptivePackedRuleTable>> {
        match &self.merges
        {
            RuleTable::Compact(rules) => Some(rules),
            RuleTable::Wide(_) => None,
        }
    }

    /// Encodes one complete pre-tokenized piece.
    ///
    /// Compact execution is selected only when every input token and local node
    /// index fits its checked `u32` domain. Otherwise the historical wide path
    /// runs the entire piece; no chunking or partial compact execution occurs.
    pub fn encode_ids(&self, input: &[TokenId]) -> Vec<TokenId> {
        if input.len() < 2 || self.merges.is_empty()
        {
            return input.to_vec();
        }
        if self.can_encode_compact(input)
        {
            self.encode_ids_compact(input)
        }
        else
        {
            self.encode_ids_wide(input)
        }
    }

    fn can_encode_compact(&self, input: &[TokenId]) -> bool {
        self.merges.is_compact()
            && try_compact_index(input.len() - 1).is_ok()
            && input.iter().all(|&token| u32::try_from(token).is_ok())
    }

    fn encode_ids_compact(&self, input: &[TokenId]) -> Vec<TokenId> {
        let mut nodes = Vec::with_capacity(input.len());
        for (index, &token) in input.iter().enumerate()
        {
            let index = try_compact_index(index).expect("compact input preflight checked indices");
            let token = u32::try_from(token).expect("compact input preflight checked token ids");
            nodes.push(CompactNode {
                token,
                prev: if index == 0
                {
                    COMPACT_INDEX_NONE
                }
                else
                {
                    index - 1
                },
                next: if usize::try_from(index).expect("compact index fits usize") + 1 < input.len()
                {
                    index + 1
                }
                else
                {
                    COMPACT_INDEX_NONE
                },
                generation: 0,
            });
        }

        let mut candidates = Vec::with_capacity(input.len().saturating_sub(1));
        for left in 0..input.len() - 1
        {
            let left = try_compact_index(left).expect("compact input preflight checked indices");
            self.push_compact_candidate(&nodes, left, left + 1, &mut candidates);
        }

        loop
        {
            candidates.retain(|candidate| Self::compact_candidate_is_valid(&nodes, *candidate));
            let Some(best_index) = candidates
                .iter()
                .enumerate()
                .min_by_key(|(_, candidate)| candidate.priority)
                .map(|(index, _)| index)
            else
            {
                break;
            };

            let candidate = candidates.swap_remove(best_index);
            let left = candidate.priority.left_index();
            let right = candidate.right;
            let left_index = usize::try_from(left).expect("compact index fits usize");
            let right_index = usize::try_from(right).expect("compact index fits usize");
            let prev = nodes[left_index].prev;
            let next = nodes[right_index].next;

            nodes[left_index].token = candidate.output;
            nodes[left_index].next = next;
            nodes[left_index].generation = nodes[left_index].generation.wrapping_add(1);

            nodes[right_index].prev = COMPACT_INDEX_INACTIVE;
            nodes[right_index].next = COMPACT_INDEX_INACTIVE;
            nodes[right_index].generation = nodes[right_index].generation.wrapping_add(1);

            if next != COMPACT_INDEX_NONE
            {
                let next_index = usize::try_from(next).expect("compact index fits usize");
                nodes[next_index].prev = left;
            }
            if prev != COMPACT_INDEX_NONE
            {
                self.push_compact_candidate(&nodes, prev, left, &mut candidates);
            }
            if next != COMPACT_INDEX_NONE
            {
                self.push_compact_candidate(&nodes, left, next, &mut candidates);
            }
        }

        let mut output = Vec::with_capacity(input.len());
        let mut current = 0u32;
        while current != COMPACT_INDEX_NONE
        {
            let index = usize::try_from(current).expect("compact index fits usize");
            output.push(usize::try_from(nodes[index].token).expect("u32 token id fits usize"));
            current = nodes[index].next;
        }
        output
    }

    fn push_compact_candidate(
        &self,
        nodes: &[CompactNode],
        left: u32,
        right: u32,
        candidates: &mut Vec<CompactCandidate>,
    ) {
        let left_index = usize::try_from(left).expect("compact index fits usize");
        let right_index = usize::try_from(right).expect("compact index fits usize");
        let Some(rule) = self
            .merges
            .get_compact(nodes[left_index].token, nodes[right_index].token)
        else
        {
            return;
        };
        candidates.push(CompactCandidate {
            priority: PriorityKey::new(rule.rank(), left),
            right,
            left_generation: nodes[left_index].generation,
            right_generation: nodes[right_index].generation,
            output: rule.output(),
        });
    }

    fn compact_candidate_is_valid(nodes: &[CompactNode], candidate: CompactCandidate) -> bool {
        let left = candidate.priority.left_index();
        let Ok(left_index) = usize::try_from(left)
        else
        {
            return false;
        };
        let Ok(right_index) = usize::try_from(candidate.right)
        else
        {
            return false;
        };
        let Some(left_node) = nodes.get(left_index)
        else
        {
            return false;
        };
        let Some(right_node) = nodes.get(right_index)
        else
        {
            return false;
        };
        left_node.is_active()
            && right_node.is_active()
            && left_node.next == candidate.right
            && right_node.prev == left
            && left_node.generation == candidate.left_generation
            && right_node.generation == candidate.right_generation
    }

    fn encode_ids_wide(&self, input: &[TokenId]) -> Vec<TokenId> {
        let mut nodes: Vec<WideNode> = input
            .iter()
            .copied()
            .enumerate()
            .map(|(index, token)| WideNode {
                token,
                prev: index.checked_sub(1),
                next: (index + 1 < input.len()).then_some(index + 1),
                generation: 0,
                active: true,
            })
            .collect();

        let mut candidates = Vec::with_capacity(input.len().saturating_sub(1));
        for left in 0..input.len() - 1
        {
            self.push_wide_candidate(&nodes, left, left + 1, &mut candidates);
        }

        loop
        {
            candidates.retain(|candidate| Self::wide_candidate_is_valid(&nodes, *candidate));
            let Some(best_index) = candidates
                .iter()
                .enumerate()
                .min_by_key(|(_, candidate)| (candidate.rank, candidate.left))
                .map(|(index, _)| index)
            else
            {
                break;
            };

            let candidate = candidates.swap_remove(best_index);
            let left = candidate.left;
            let right = candidate.right;
            let prev = nodes[left].prev;
            let next = nodes[right].next;

            nodes[left].token = candidate.output;
            nodes[left].next = next;
            nodes[left].generation = nodes[left].generation.wrapping_add(1);

            nodes[right].active = false;
            nodes[right].generation = nodes[right].generation.wrapping_add(1);

            if let Some(next_index) = next
            {
                nodes[next_index].prev = Some(left);
            }
            if let Some(prev_index) = prev
            {
                self.push_wide_candidate(&nodes, prev_index, left, &mut candidates);
            }
            if let Some(next_index) = next
            {
                self.push_wide_candidate(&nodes, left, next_index, &mut candidates);
            }
        }

        let mut output = Vec::with_capacity(input.len());
        let mut current = Some(0usize);
        while let Some(index) = current
        {
            if nodes[index].active
            {
                output.push(nodes[index].token);
            }
            current = nodes[index].next;
        }
        output
    }

    fn push_wide_candidate(
        &self,
        nodes: &[WideNode],
        left: usize,
        right: usize,
        candidates: &mut Vec<WideCandidate>,
    ) {
        let Some(rule) = self.merges.get(nodes[left].token, nodes[right].token)
        else
        {
            return;
        };
        candidates.push(WideCandidate {
            rank: rule.rank,
            left,
            right,
            left_generation: nodes[left].generation,
            right_generation: nodes[right].generation,
            output: rule.output,
        });
    }

    fn wide_candidate_is_valid(nodes: &[WideNode], candidate: WideCandidate) -> bool {
        let Some(left) = nodes.get(candidate.left)
        else
        {
            return false;
        };
        let Some(right) = nodes.get(candidate.right)
        else
        {
            return false;
        };
        left.active
            && right.active
            && left.next == Some(candidate.right)
            && right.prev == Some(candidate.left)
            && left.generation == candidate.left_generation
            && right.generation == candidate.right_generation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elastic_tokenizer::CanonicalBpeOracle;

    fn assert_parity(merges: &[(TokenId, TokenId, TokenId)], input: &[TokenId]) {
        let reference = CanonicalBpeOracle::from_ordered_merges(merges).unwrap();
        let indexed = IndexedBpe::from_ordered_merges(merges).unwrap();
        assert_eq!(indexed.encode_ids(input), reference.encode_ids(input));
    }

    #[test]
    fn compact_layout_reduces_working_set() {
        assert_eq!(std::mem::size_of::<CompactNode>(), 16);
        assert_eq!(std::mem::size_of::<CompactCandidate>(), 24);
        if usize::BITS == 64
        {
            assert!(std::mem::size_of::<CompactNode>() < std::mem::size_of::<WideNode>());
            assert!(std::mem::size_of::<CompactCandidate>() < std::mem::size_of::<WideCandidate>());
        }
    }

    #[test]
    fn normal_rule_tables_use_compact_words() {
        let indexed = IndexedBpe::from_ordered_merges(&[(1, 2, 3), (3, 4, 5)]).unwrap();
        assert!(matches!(indexed.merges, RuleTable::Compact(_)));
    }

    #[test]
    fn indexed_matches_rank_priority_conflict() {
        let merges = [(2, 3, 10), (1, 2, 11)];
        assert_parity(&merges, &[1, 2, 3]);
    }

    #[test]
    fn indexed_matches_recursive_and_overlapping_merges() {
        let merges = [
            (2, 3, 10),
            (1, 2, 11),
            (1, 10, 12),
            (11, 3, 13),
            (3, 3, 14),
            (14, 1, 15),
        ];
        assert_parity(&merges, &[1, 2, 3, 3, 1]);
    }

    #[test]
    fn indexed_exhaustive_small_alphabet_parity() {
        let merges = [
            (2, 3, 10),
            (1, 2, 11),
            (1, 10, 12),
            (11, 3, 13),
            (3, 3, 14),
            (14, 1, 15),
            (10, 14, 16),
        ];
        let reference = CanonicalBpeOracle::from_ordered_merges(&merges).unwrap();
        let indexed = IndexedBpe::from_ordered_merges(&merges).unwrap();

        for len in 0..=8
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
                    indexed.encode_ids(&input),
                    reference.encode_ids(&input),
                    "indexed parity failed for input {input:?}"
                );
            }
        }
    }

    #[test]
    fn indexed_matches_reference_on_long_repetitive_piece() {
        let merges = [(1, 1, 2), (2, 2, 3), (3, 3, 4), (4, 4, 5)];
        let input = vec![1; 1024];
        assert_parity(&merges, &input);
    }

    #[test]
    fn wide_rule_fallback_preserves_semantics() {
        if usize::BITS > 32
        {
            let wide = usize::try_from(u64::from(u32::MAX) + 1).unwrap();
            let indexed = IndexedBpe::from_ordered_merges(&[(wide, 1, 2)]).unwrap();
            assert!(matches!(indexed.merges, RuleTable::Wide(_)));
            assert!(!indexed.can_encode_compact(&[wide, 1]));
            assert_eq!(indexed.encode_ids(&[wide, 1]), vec![2]);
        }
    }

    #[test]
    fn duplicate_rules_are_rejected_consistently() {
        let err = IndexedBpe::from_ordered_merges(&[(1, 2, 3), (1, 2, 4)]).unwrap_err();
        assert_eq!(err, DuplicateMergeRule { left: 1, right: 2 });
    }
}
