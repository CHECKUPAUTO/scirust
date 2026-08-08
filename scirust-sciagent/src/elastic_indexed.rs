//! Indexed medium-piece BPE kernel for the ElasticTokenizer.
//!
//! `IndexedBpe` keeps the piece as an indexed linked list. Merge candidates
//! carry node generations so only adjacencies touched by a merge need to be
//! regenerated. Candidate selection is intentionally linear in this phase;
//! the later `Heap` kernel will replace that scheduler for large pieces while
//! preserving the exact same rank-priority semantics.

use std::collections::BTreeMap;

use crate::elastic_tokenizer::{DuplicateMergeRule, TokenId};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct IndexedRule {
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
struct Candidate {
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
    merges: BTreeMap<(TokenId, TokenId), IndexedRule>,
}

impl IndexedBpe {
    pub fn from_ordered_merges(
        merges: &[(TokenId, TokenId, TokenId)],
    ) -> Result<Self, DuplicateMergeRule> {
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
        Ok(Self { merges: ranked })
    }

    /// Encodes one complete pre-tokenized piece.
    ///
    /// Original node indices remain monotonic from left to right because a
    /// merge always keeps the left node. They therefore provide the exact
    /// left-most tie break required when the same ranked rule occurs more than
    /// once in the current piece.
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

        let mut candidates = Vec::with_capacity(input.len().saturating_sub(1));
        for left in 0..input.len() - 1
        {
            self.push_candidate(&nodes, left, left + 1, &mut candidates);
        }

        loop
        {
            candidates.retain(|candidate| Self::candidate_is_valid(&nodes, *candidate));

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
                self.push_candidate(&nodes, prev_index, left, &mut candidates);
            }
            if let Some(next_index) = next
            {
                self.push_candidate(&nodes, left, next_index, &mut candidates);
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
        candidates: &mut Vec<Candidate>,
    ) {
        let Some(rule) = self.merges.get(&(nodes[left].token, nodes[right].token))
        else
        {
            return;
        };
        candidates.push(Candidate {
            rank: rule.rank,
            left,
            right,
            left_generation: nodes[left].generation,
            right_generation: nodes[right].generation,
            output: rule.output,
        });
    }

    fn candidate_is_valid(nodes: &[Node], candidate: Candidate) -> bool {
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
    fn duplicate_rules_are_rejected_consistently() {
        let err = IndexedBpe::from_ordered_merges(&[(1, 2, 3), (1, 2, 4)]).unwrap_err();
        assert_eq!(err, DuplicateMergeRule { left: 1, right: 2 });
    }
}
