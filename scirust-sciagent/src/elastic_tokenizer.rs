//! Semantic core for the SciAgent ElasticTokenizer.
//!
//! The central contract is deliberately stronger than a performance promise:
//! routing may change *how* a BPE piece is encoded, but it must never change
//! *which token ids* are produced.  Piece classes are therefore execution
//! classes only.  They are not artificial BPE chunk boundaries.
//!
//! This module starts with a deliberately simple O(n²) rank-priority oracle.
//! Optimized S/M/L/XL/XXL/XXXL kernels must prove exact parity with this oracle
//! before they can be selected by an [`ElasticProfile`].

use std::collections::BTreeMap;
use std::fmt;

/// Token identifier used by the current SciAgent BPE implementation.
pub type TokenId = usize;

/// Execution-size class selected by the elastic router.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PieceClass {
    S,
    M,
    L,
    Xl,
    Xxl,
    Xxxl,
}

impl PieceClass {
    const fn index(self) -> usize {
        match self {
            Self::S => 0,
            Self::M => 1,
            Self::L => 2,
            Self::Xl => 3,
            Self::Xxl => 4,
            Self::Xxxl => 5,
        }
    }
}

/// BPE execution kernel.
///
/// Only [`Self::Reference`] is semantically implemented in the first phase.
/// The remaining variants reserve stable profile names for the optimized
/// kernels that will be added and parity-gated in subsequent phases.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BpeKernel {
    Reference,
    TinyScan,
    Indexed,
    Heap,
}

/// Monotonic byte-length boundaries for the six execution classes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ElasticThresholds {
    pub s_max: usize,
    pub m_max: usize,
    pub l_max: usize,
    pub xl_max: usize,
    pub xxl_max: usize,
}

impl ElasticThresholds {
    /// Creates validated monotonic thresholds.
    pub fn new(
        s_max: usize,
        m_max: usize,
        l_max: usize,
        xl_max: usize,
        xxl_max: usize,
    ) -> Result<Self, ThresholdError> {
        if !(s_max < m_max && m_max < l_max && l_max < xl_max && xl_max < xxl_max) {
            return Err(ThresholdError);
        }
        Ok(Self {
            s_max,
            m_max,
            l_max,
            xl_max,
            xxl_max,
        })
    }

    /// Classifies one complete BPE piece by byte length.
    ///
    /// Classification never splits the piece.  This matters because a valid
    /// merge may cross any arbitrary internal byte offset.
    pub fn classify(self, piece_len: usize) -> PieceClass {
        if piece_len <= self.s_max {
            PieceClass::S
        } else if piece_len <= self.m_max {
            PieceClass::M
        } else if piece_len <= self.l_max {
            PieceClass::L
        } else if piece_len <= self.xl_max {
            PieceClass::Xl
        } else if piece_len <= self.xxl_max {
            PieceClass::Xxl
        } else {
            PieceClass::Xxxl
        }
    }
}

/// Invalid elastic thresholds (they must be strictly increasing).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThresholdError;

impl fmt::Display for ThresholdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("elastic tokenizer thresholds must be strictly increasing")
    }
}

impl std::error::Error for ThresholdError {}

/// Hardware-local execution profile.
///
/// A profile is explicitly not part of tokenizer semantics. Two hosts may use
/// different profiles and still must produce identical token ids.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ElasticProfile {
    thresholds: ElasticThresholds,
    kernels: [BpeKernel; 6],
}

impl ElasticProfile {
    pub const fn new(thresholds: ElasticThresholds, kernels: [BpeKernel; 6]) -> Self {
        Self {
            thresholds,
            kernels,
        }
    }

    /// Safe bootstrap profile used before auto-calibration is available.
    pub const fn reference_only(thresholds: ElasticThresholds) -> Self {
        Self::new(thresholds, [BpeKernel::Reference; 6])
    }

    pub fn class_for(self, piece_len: usize) -> PieceClass {
        self.thresholds.classify(piece_len)
    }

    pub fn kernel_for(self, piece_len: usize) -> BpeKernel {
        self.kernels[self.class_for(piece_len).index()]
    }

    pub const fn thresholds(self) -> ElasticThresholds {
        self.thresholds
    }

    pub const fn kernels(self) -> [BpeKernel; 6] {
        self.kernels
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RankedMerge {
    output: TokenId,
    rank: usize,
}

/// Error returned when an ordered merge table assigns the same input pair more
/// than once. Such a table has ambiguous rank semantics and is rejected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DuplicateMergeRule {
    pub left: TokenId,
    pub right: TokenId,
}

impl fmt::Display for DuplicateMergeRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "duplicate BPE merge rule for pair ({}, {})",
            self.left, self.right
        )
    }
}

impl std::error::Error for DuplicateMergeRule {}

/// Canonical rank-priority BPE oracle over an already pre-tokenized piece.
///
/// Rank is the position in the ordered merge table, independent of output token
/// id. On each iteration the globally lowest-rank adjacent merge wins; equal
/// ranks are resolved by the left-most occurrence. The implementation is kept
/// intentionally simple and auditable rather than fast.
#[derive(Clone, Debug)]
pub struct CanonicalBpeOracle {
    merges: BTreeMap<(TokenId, TokenId), RankedMerge>,
}

impl CanonicalBpeOracle {
    /// Builds the semantic oracle from merge rules ordered from highest to
    /// lowest priority.
    pub fn from_ordered_merges(
        merges: &[(TokenId, TokenId, TokenId)],
    ) -> Result<Self, DuplicateMergeRule> {
        let mut ranked = BTreeMap::new();
        for (rank, &(left, right, output)) in merges.iter().enumerate() {
            if ranked
                .insert((left, right), RankedMerge { output, rank })
                .is_some()
            {
                return Err(DuplicateMergeRule { left, right });
            }
        }
        Ok(Self { merges: ranked })
    }

    pub fn is_empty(&self) -> bool {
        self.merges.is_empty()
    }

    /// Encodes one complete piece according to canonical BPE rank priority.
    ///
    /// This function never chunks `input`; every adjacent boundary remains
    /// visible to the oracle for the whole reduction.
    pub fn encode_ids(&self, input: &[TokenId]) -> Vec<TokenId> {
        let mut ids = input.to_vec();

        while ids.len() >= 2 {
            let mut best: Option<(usize, usize, TokenId)> = None;

            for (position, pair) in ids.windows(2).enumerate() {
                let Some(rule) = self.merges.get(&(pair[0], pair[1])) else {
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
            ids[position] = output;
            ids.remove(position + 1);
        }

        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_oracle_respects_rank_over_left_to_right_order() {
        // Input: a b c.  `b+c` has the better rank even though `a+b` appears
        // earlier in the piece. A left-to-right bulk merge would incorrectly
        // produce [ab, c]. Canonical BPE must produce [a, bc].
        const A: TokenId = 10;
        const B: TokenId = 11;
        const C: TokenId = 12;
        const BC: TokenId = 20;
        const AB: TokenId = 21;

        let oracle = CanonicalBpeOracle::from_ordered_merges(&[(B, C, BC), (A, B, AB)]).unwrap();
        assert_eq!(oracle.encode_ids(&[A, B, C]), vec![A, BC]);
    }

    #[test]
    fn canonical_oracle_uses_leftmost_occurrence_for_same_rule() {
        const A: TokenId = 10;
        const AA: TokenId = 20;

        let oracle = CanonicalBpeOracle::from_ordered_merges(&[(A, A, AA)]).unwrap();
        assert_eq!(oracle.encode_ids(&[A, A, A]), vec![AA, A]);
    }

    #[test]
    fn canonical_oracle_rejects_duplicate_pair_rules() {
        let err = CanonicalBpeOracle::from_ordered_merges(&[(1, 2, 3), (1, 2, 4)]).unwrap_err();
        assert_eq!(err, DuplicateMergeRule { left: 1, right: 2 });
    }

    #[test]
    fn thresholds_cover_all_six_classes_without_chunking() {
        let thresholds = ElasticThresholds::new(16, 64, 256, 1024, 4096).unwrap();
        assert_eq!(thresholds.classify(0), PieceClass::S);
        assert_eq!(thresholds.classify(16), PieceClass::S);
        assert_eq!(thresholds.classify(17), PieceClass::M);
        assert_eq!(thresholds.classify(65), PieceClass::L);
        assert_eq!(thresholds.classify(257), PieceClass::Xl);
        assert_eq!(thresholds.classify(1025), PieceClass::Xxl);
        assert_eq!(thresholds.classify(4097), PieceClass::Xxxl);
    }

    #[test]
    fn profile_routes_classes_without_affecting_semantics() {
        let thresholds = ElasticThresholds::new(16, 64, 256, 1024, 4096).unwrap();
        let profile = ElasticProfile::new(
            thresholds,
            [
                BpeKernel::TinyScan,
                BpeKernel::TinyScan,
                BpeKernel::Indexed,
                BpeKernel::Indexed,
                BpeKernel::Heap,
                BpeKernel::Heap,
            ],
        );
        assert_eq!(profile.kernel_for(8), BpeKernel::TinyScan);
        assert_eq!(profile.kernel_for(128), BpeKernel::Indexed);
        assert_eq!(profile.kernel_for(8192), BpeKernel::Heap);
    }

    #[test]
    fn thresholds_must_be_strictly_increasing() {
        assert_eq!(
            ElasticThresholds::new(16, 16, 256, 1024, 4096),
            Err(ThresholdError)
        );
    }
}
