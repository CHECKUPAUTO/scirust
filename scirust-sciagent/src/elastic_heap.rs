//! Large-piece BPE kernel for the ElasticTokenizer.
//!
//! `HeapBpe` keeps the piece as an indexed linked list and schedules valid
//! merge candidates in a binary heap. Node generations make stale candidates
//! cheap to reject after a local merge, so large and pathological pieces avoid
//! rescanning every candidate on each reduction step.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};

use crate::elastic_tokenizer::{DuplicateMergeRule, TokenId};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HeapRule {
    output: TokenId,
    rank: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Node {
    token: TokenId,
    prev: Option<usize>,
    next: Option<usize>,
    generation: u32,
    active: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HeapCandidate {
    rank: usize,
    left: usize,
    right: usize,
    left_generation: u32,
    right_generation: u32,
    output: TokenId,
}

impl Ord for HeapCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // `BinaryHeap` is a max-heap. Reverse the semantic key so the globally
        // lowest merge rank wins, with the left-most active occurrence as the
        // deterministic tie break. Remaining fields make the ordering total.
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

impl PartialOrd for HeapCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Rank-priority BPE kernel with heap-scheduled local candidates.
#[derive(Clone, Debug)]
pub struct HeapBpe {
    merges: BTreeMap<(TokenId, TokenId), HeapRule>,
}

impl HeapBpe {
    pub fn from_ordered_merges(
        merges: &[(TokenId, TokenId, TokenId)],
    ) -> Result<Self, DuplicateMergeRule> {
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
        Ok(Self { merges: ranked })
    }

    /// Encodes one complete pre-tokenized piece without artificial chunking.
    pub fn encode_ids(&self, input: &[TokenId]) -> Vec<TokenId> {
        if input.len() < 2 || self.merges.is_empty()
        {
            return input.to_vec();
        }

        let mut nodes: Vec<Node> = input
            .iter()
            .copied()
            .enumerate()
            .map(|(index, token)| Node {
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
            self.push_candidate(&nodes, left, left + 1, &mut heap);
        }

        while let Some(candidate) = self.pop_valid_candidate(&nodes, &mut heap)
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
                self.push_candidate(&nodes, prev_index, left, &mut heap);
            }
            if let Some(next_index) = next
            {
                self.push_candidate(&nodes, left, next_index, &mut heap);
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

    fn push_candidate(
        &self,
        nodes: &[Node],
        left: usize,
        right: usize,
        heap: &mut BinaryHeap<HeapCandidate>,
    ) {
        let Some(rule) = self.merges.get(&(nodes[left].token, nodes[right].token))
        else
        {
            return;
        };
        heap.push(HeapCandidate {
            rank: rule.rank,
            left,
            right,
            left_generation: nodes[left].generation,
            right_generation: nodes[right].generation,
            output: rule.output,
        });
    }

    fn pop_valid_candidate(
        &self,
        nodes: &[Node],
        heap: &mut BinaryHeap<HeapCandidate>,
    ) -> Option<HeapCandidate> {
        while let Some(candidate) = heap.pop()
        {
            if Self::candidate_is_valid(nodes, candidate)
            {
                return Some(candidate);
            }
        }
        None
    }

    fn candidate_is_valid(nodes: &[Node], candidate: HeapCandidate) -> bool {
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
        let merges = [
            (1, 1, 2),
            (2, 2, 3),
            (3, 3, 4),
            (4, 4, 5),
            (5, 5, 6),
        ];
        let input = vec![1; 8192];
        assert_parity(&merges, &input);
    }

    #[test]
    fn duplicate_rules_are_rejected_consistently() {
        let err = HeapBpe::from_ordered_merges(&[(1, 2, 3), (1, 2, 4)]).unwrap_err();
        assert_eq!(err, DuplicateMergeRule { left: 1, right: 2 });
    }
}
