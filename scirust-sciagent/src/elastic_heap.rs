//! Large-piece BPE kernel for the ElasticTokenizer.
//!
//! `HeapBpe` keeps the piece as an indexed linked list and schedules valid
//! merge candidates in a binary heap. Normal pieces use compact `u32` nodes and
//! a packed `(rank, left-index)` priority; a wide path preserves compatibility
//! for inputs outside the compact domains.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};

use crate::elastic_id::{
    try_compact_index, COMPACT_INDEX_INACTIVE, COMPACT_INDEX_NONE, PackedRule, PairKey,
    PriorityKey,
};
use crate::elastic_tokenizer::{DuplicateMergeRule, TokenId};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HeapRule {
    output: TokenId,
    rank: usize,
}

#[derive(Clone, Debug)]
enum RuleTable {
    Compact(BTreeMap<PairKey, PackedRule>),
    Wide(BTreeMap<(TokenId, TokenId), HeapRule>),
}

impl RuleTable {
    fn from_ordered_merges(
        merges: &[(TokenId, TokenId, TokenId)],
    ) -> Result<Self, DuplicateMergeRule> {
        let compact = merges
            .iter()
            .enumerate()
            .all(|(rank, &(left, right, output))| {
                u32::try_from(rank).is_ok()
                    && u32::try_from(left).is_ok()
                    && u32::try_from(right).is_ok()
                    && u32::try_from(output).is_ok()
            });

        if compact
        {
            let mut ranked = BTreeMap::new();
            for (rank, &(left, right, output)) in merges.iter().enumerate()
            {
                let key = PairKey::try_from_usize(left, right)
                    .expect("compact table preflight checked token ids");
                let rule = PackedRule::try_from_usize(rank, output)
                    .expect("compact table preflight checked rule fields");
                if ranked.insert(key, rule).is_some()
                {
                    return Err(DuplicateMergeRule { left, right });
                }
            }
            Ok(Self::Compact(ranked))
        }
        else
        {
            let mut ranked = BTreeMap::new();
            for (rank, &(left, right, output)) in merges.iter().enumerate()
            {
                if ranked
                    .insert((left, right), HeapRule { output, rank })
                    .is_some()
                {
                    return Err(DuplicateMergeRule { left, right });
                }
            }
            Ok(Self::Wide(ranked))
        }
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
            Self::Compact(rules) => rules.get(&PairKey::new(left, right)).copied(),
            Self::Wide(_) => None,
        }
    }

    fn get(&self, left: TokenId, right: TokenId) -> Option<HeapRule> {
        match self
        {
            Self::Compact(rules) =>
            {
                let key = PairKey::try_from_usize(left, right).ok()?;
                let rule = rules.get(&key)?;
                Some(HeapRule {
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
struct CompactHeapCandidate {
    priority: PriorityKey,
    right: u32,
    left_generation: u32,
    right_generation: u32,
    output: u32,
}

impl Ord for CompactHeapCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .priority
            .cmp(&self.priority)
            .then_with(|| other.right.cmp(&self.right))
            .then_with(|| other.left_generation.cmp(&self.left_generation))
            .then_with(|| other.right_generation.cmp(&self.right_generation))
            .then_with(|| other.output.cmp(&self.output))
    }
}

impl PartialOrd for CompactHeapCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
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
struct WideHeapCandidate {
    rank: usize,
    left: usize,
    right: usize,
    left_generation: u32,
    right_generation: u32,
    output: TokenId,
}

impl Ord for WideHeapCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.left.cmp(&self.left))
            .then_with(|| other.right.cmp(&self.right))
            .then_with(|| other.left_generation.cmp(&self.left_generation))
            .then_with(|| other.right_generation.cmp(&self.right_generation))
            .then_with(|| other.output.cmp(&self.output))
    }
}

impl PartialOrd for WideHeapCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Rank-priority BPE kernel with heap-scheduled local candidates.
#[derive(Clone, Debug)]
pub struct HeapBpe {
    merges: RuleTable,
}

impl HeapBpe {
    pub fn from_ordered_merges(
        merges: &[(TokenId, TokenId, TokenId)],
    ) -> Result<Self, DuplicateMergeRule> {
        Ok(Self {
            merges: RuleTable::from_ordered_merges(merges)?,
        })
    }

    /// Encodes one complete pre-tokenized piece without artificial chunking.
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
                next: if usize::try_from(index).expect("compact index fits usize") + 1
                    < input.len()
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

        let mut heap = BinaryHeap::with_capacity(input.len().saturating_sub(1));
        for left in 0..input.len() - 1
        {
            let left = try_compact_index(left).expect("compact input preflight checked indices");
            self.push_compact_candidate(&nodes, left, left + 1, &mut heap);
        }

        while let Some(candidate) = Self::pop_valid_compact_candidate(&nodes, &mut heap)
        {
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
                self.push_compact_candidate(&nodes, prev, left, &mut heap);
            }
            if next != COMPACT_INDEX_NONE
            {
                self.push_compact_candidate(&nodes, left, next, &mut heap);
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
        heap: &mut BinaryHeap<CompactHeapCandidate>,
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
        heap.push(CompactHeapCandidate {
            priority: PriorityKey::new(rule.rank(), left),
            right,
            left_generation: nodes[left_index].generation,
            right_generation: nodes[right_index].generation,
            output: rule.output(),
        });
    }

    fn pop_valid_compact_candidate(
        nodes: &[CompactNode],
        heap: &mut BinaryHeap<CompactHeapCandidate>,
    ) -> Option<CompactHeapCandidate> {
        while let Some(candidate) = heap.pop()
        {
            if Self::compact_candidate_is_valid(nodes, candidate)
            {
                return Some(candidate);
            }
        }
        None
    }

    fn compact_candidate_is_valid(
        nodes: &[CompactNode],
        candidate: CompactHeapCandidate,
    ) -> bool {
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

        let mut heap = BinaryHeap::with_capacity(input.len().saturating_sub(1));
        for left in 0..input.len() - 1
        {
            self.push_wide_candidate(&nodes, left, left + 1, &mut heap);
        }

        while let Some(candidate) = Self::pop_valid_wide_candidate(&nodes, &mut heap)
        {
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
                self.push_wide_candidate(&nodes, prev_index, left, &mut heap);
            }
            if let Some(next_index) = next
            {
                self.push_wide_candidate(&nodes, left, next_index, &mut heap);
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
        heap: &mut BinaryHeap<WideHeapCandidate>,
    ) {
        let Some(rule) = self.merges.get(nodes[left].token, nodes[right].token)
        else
        {
            return;
        };
        heap.push(WideHeapCandidate {
            rank: rule.rank,
            left,
            right,
            left_generation: nodes[left].generation,
            right_generation: nodes[right].generation,
            output: rule.output,
        });
    }

    fn pop_valid_wide_candidate(
        nodes: &[WideNode],
        heap: &mut BinaryHeap<WideHeapCandidate>,
    ) -> Option<WideHeapCandidate> {
        while let Some(candidate) = heap.pop()
        {
            if Self::wide_candidate_is_valid(nodes, candidate)
            {
                return Some(candidate);
            }
        }
        None
    }

    fn wide_candidate_is_valid(nodes: &[WideNode], candidate: WideHeapCandidate) -> bool {
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
        let heap = HeapBpe::from_ordered_merges(merges).unwrap();
        assert_eq!(heap.encode_ids(input), reference.encode_ids(input));
    }

    #[test]
    fn compact_layout_reduces_working_set() {
        assert_eq!(std::mem::size_of::<CompactNode>(), 16);
        assert_eq!(std::mem::size_of::<CompactHeapCandidate>(), 24);
        if usize::BITS == 64
        {
            assert!(std::mem::size_of::<CompactNode>() < std::mem::size_of::<WideNode>());
            assert!(
                std::mem::size_of::<CompactHeapCandidate>()
                    < std::mem::size_of::<WideHeapCandidate>()
            );
        }
    }

    #[test]
    fn normal_rule_tables_use_compact_words() {
        let heap = HeapBpe::from_ordered_merges(&[(1, 2, 3), (3, 4, 5)]).unwrap();
        assert!(matches!(heap.merges, RuleTable::Compact(_)));
    }

    #[test]
    fn heap_matches_rank_priority_conflict() {
        let merges = [(2, 3, 10), (1, 2, 11)];
        assert_parity(&merges, &[1, 2, 3]);
    }

    #[test]
    fn heap_uses_leftmost_occurrence_for_equal_rank() {
        let merges = [(1, 1, 2)];
        assert_parity(&merges, &[1, 1, 1, 1, 1]);
    }

    #[test]
    fn heap_matches_recursive_and_overlapping_merges() {
        let merges = [
            (2, 3, 10),
            (1, 2, 11),
            (1, 10, 12),
            (11, 3, 13),
            (3, 3, 14),
            (14, 1, 15),
            (10, 14, 16),
        ];
        assert_parity(&merges, &[1, 2, 3, 3, 1]);
    }

    #[test]
    fn heap_exhaustive_small_alphabet_parity() {
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
        let heap = HeapBpe::from_ordered_merges(&merges).unwrap();

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
                    heap.encode_ids(&input),
                    reference.encode_ids(&input),
                    "heap parity failed for input {input:?}"
                );
            }
        }
    }

    #[test]
    fn heap_matches_reference_on_pathological_repetitive_piece() {
        let merges = [(1, 1, 2), (2, 2, 3), (3, 3, 4), (4, 4, 5), (5, 5, 6)];
        let input = vec![1; 8192];
        assert_parity(&merges, &input);
    }

    #[test]
    fn wide_rule_fallback_preserves_semantics() {
        if usize::BITS > 32
        {
            let wide = usize::try_from(u64::from(u32::MAX) + 1).unwrap();
            let heap = HeapBpe::from_ordered_merges(&[(wide, 1, 2)]).unwrap();
            assert!(matches!(heap.merges, RuleTable::Wide(_)));
            assert!(!heap.can_encode_compact(&[wide, 1]));
            assert_eq!(heap.encode_ids(&[wide, 1]), vec![2]);
        }
    }

    #[test]
    fn duplicate_rules_are_rejected_consistently() {
        let err = HeapBpe::from_ordered_merges(&[(1, 2, 3), (1, 2, 4)]).unwrap_err();
        assert_eq!(err, DuplicateMergeRule { left: 1, right: 2 });
    }
}
