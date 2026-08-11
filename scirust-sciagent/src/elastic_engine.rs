//! Semantics-safe execution router for ElasticTokenizer BPE kernels.
//!
//! The router records both the requested and executed kernels. Out-of-capacity
//! tiny pieces fall back to the canonical oracle; routing never changes
//! tokenization and never splits a piece.

use std::sync::Arc;

use crate::elastic_heap::HeapBpe;
use crate::elastic_indexed::IndexedBpe;
use crate::elastic_rule_table::AdaptivePackedRuleTable;
use crate::elastic_tiny::TinyScanBpe;
use crate::elastic_tokenizer::{
    BpeKernel, CanonicalBpeOracle, DuplicateMergeRule, ElasticProfile, PieceClass, TokenId,
};

/// Result of one elastic BPE piece encoding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ElasticEncoding {
    pub ids: Vec<TokenId>,
    pub class: PieceClass,
    pub requested_kernel: BpeKernel,
    pub executed_kernel: BpeKernel,
}

/// BPE engine that routes complete pieces while preserving one canonical
/// rank-priority semantic contract.
#[derive(Clone, Debug)]
pub struct ElasticBpeEngine {
    reference: CanonicalBpeOracle,
    tiny_scan: TinyScanBpe,
    indexed: IndexedBpe,
    heap: HeapBpe,
    profile: ElasticProfile,
}

impl ElasticBpeEngine {
    pub fn from_ordered_merges(
        merges: &[(TokenId, TokenId, TokenId)],
        profile: ElasticProfile,
    ) -> Result<Self, DuplicateMergeRule> {
        let reference = CanonicalBpeOracle::from_ordered_merges(merges)?;
        let shared_rules = AdaptivePackedRuleTable::try_from_ordered_merges(merges)?;
        let (tiny_scan, indexed, heap) = if let Some(rules) = shared_rules
        {
            let rules = Arc::new(rules);
            (
                TinyScanBpe::from_shared_compact_rules(Arc::clone(&rules)),
                IndexedBpe::from_shared_compact_rules(Arc::clone(&rules)),
                HeapBpe::from_shared_compact_rules(rules),
            )
        }
        else
        {
            (
                TinyScanBpe::from_ordered_merges(merges)?,
                IndexedBpe::from_ordered_merges(merges)?,
                HeapBpe::from_ordered_merges(merges)?,
            )
        };

        Ok(Self {
            reference,
            tiny_scan,
            indexed,
            heap,
            profile,
        })
    }

    pub const fn profile(&self) -> ElasticProfile {
        self.profile
    }

    pub fn set_profile(&mut self, profile: ElasticProfile) {
        self.profile = profile;
    }

    /// Encodes one complete pre-tokenized piece.
    ///
    /// `piece_len` is the original byte length used for profile routing. `input`
    /// remains whole regardless of the selected execution class.
    pub fn encode_ids(&self, input: &[TokenId], piece_len: usize) -> ElasticEncoding {
        let class = self.profile.class_for(piece_len);
        let requested_kernel = self.profile.kernel_for(piece_len);

        let (ids, executed_kernel) = match requested_kernel
        {
            BpeKernel::Reference => (self.reference.encode_ids(input), BpeKernel::Reference),
            BpeKernel::TinyScan =>
            {
                if let Some(ids) = self.tiny_scan.try_encode_ids(input)
                {
                    (ids, BpeKernel::TinyScan)
                }
                else
                {
                    (self.reference.encode_ids(input), BpeKernel::Reference)
                }
            },
            BpeKernel::Indexed => (self.indexed.encode_ids(input), BpeKernel::Indexed),
            BpeKernel::Heap => (self.heap.encode_ids(input), BpeKernel::Heap),
        };

        ElasticEncoding {
            ids,
            class,
            requested_kernel,
            executed_kernel,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elastic_tiny::TINY_SCAN_CAPACITY;
    use crate::elastic_tokenizer::ElasticThresholds;

    fn thresholds() -> ElasticThresholds {
        ElasticThresholds::new(16, 64, 256, 1024, 4096).unwrap()
    }

    #[test]
    fn router_shares_one_compact_rule_table_across_optimized_kernels() {
        let profile = ElasticProfile::reference_only(thresholds());
        let engine =
            ElasticBpeEngine::from_ordered_merges(&[(1, 2, 3), (3, 4, 5)], profile).unwrap();
        let tiny = engine.tiny_scan.compact_rule_table().unwrap();
        let indexed = engine.indexed.compact_rule_table().unwrap();
        let heap = engine.heap.compact_rule_table().unwrap();

        assert!(Arc::ptr_eq(tiny, indexed));
        assert!(Arc::ptr_eq(tiny, heap));
        assert_eq!(Arc::strong_count(tiny), 3);
    }

    #[test]
    fn router_executes_tiny_kernel_with_canonical_result() {
        let profile = ElasticProfile::new(
            thresholds(),
            [
                BpeKernel::TinyScan,
                BpeKernel::Reference,
                BpeKernel::Reference,
                BpeKernel::Reference,
                BpeKernel::Reference,
                BpeKernel::Reference,
            ],
        );
        let engine =
            ElasticBpeEngine::from_ordered_merges(&[(2, 3, 10), (1, 2, 11)], profile).unwrap();

        let encoded = engine.encode_ids(&[1, 2, 3], 3);
        assert_eq!(encoded.ids, vec![1, 10]);
        assert_eq!(encoded.class, PieceClass::S);
        assert_eq!(encoded.requested_kernel, BpeKernel::TinyScan);
        assert_eq!(encoded.executed_kernel, BpeKernel::TinyScan);
    }

    #[test]
    fn router_executes_indexed_kernel_with_canonical_result() {
        let profile = ElasticProfile::new(thresholds(), [BpeKernel::Indexed; 6]);
        let engine =
            ElasticBpeEngine::from_ordered_merges(&[(2, 3, 10), (1, 2, 11)], profile).unwrap();

        let encoded = engine.encode_ids(&[1, 2, 3], 128);
        assert_eq!(encoded.ids, vec![1, 10]);
        assert_eq!(encoded.class, PieceClass::L);
        assert_eq!(encoded.requested_kernel, BpeKernel::Indexed);
        assert_eq!(encoded.executed_kernel, BpeKernel::Indexed);
    }

    #[test]
    fn router_executes_heap_kernel_with_canonical_result() {
        let profile = ElasticProfile::new(thresholds(), [BpeKernel::Heap; 6]);
        let engine =
            ElasticBpeEngine::from_ordered_merges(&[(2, 3, 10), (1, 2, 11)], profile).unwrap();

        let encoded = engine.encode_ids(&[1, 2, 3], 8192);
        assert_eq!(encoded.ids, vec![1, 10]);
        assert_eq!(encoded.class, PieceClass::Xxxl);
        assert_eq!(encoded.requested_kernel, BpeKernel::Heap);
        assert_eq!(encoded.executed_kernel, BpeKernel::Heap);
    }

    #[test]
    fn oversized_tiny_request_falls_back_without_chunking() {
        let profile = ElasticProfile::new(thresholds(), [BpeKernel::TinyScan; 6]);
        let engine = ElasticBpeEngine::from_ordered_merges(&[(1, 1, 2)], profile).unwrap();
        let input = vec![1; TINY_SCAN_CAPACITY + 1];

        let encoded = engine.encode_ids(&input, input.len());
        let reference = CanonicalBpeOracle::from_ordered_merges(&[(1, 1, 2)]).unwrap();
        assert_eq!(encoded.ids, reference.encode_ids(&input));
        assert_eq!(encoded.requested_kernel, BpeKernel::TinyScan);
        assert_eq!(encoded.executed_kernel, BpeKernel::Reference);
    }

    #[test]
    fn profile_can_change_without_changing_token_ids() {
        let merges = [(2, 3, 10), (1, 2, 11)];
        let reference_profile = ElasticProfile::reference_only(thresholds());
        let tiny_profile = ElasticProfile::new(thresholds(), [BpeKernel::TinyScan; 6]);
        let indexed_profile = ElasticProfile::new(thresholds(), [BpeKernel::Indexed; 6]);
        let heap_profile = ElasticProfile::new(thresholds(), [BpeKernel::Heap; 6]);
        let mut engine = ElasticBpeEngine::from_ordered_merges(&merges, reference_profile).unwrap();

        let reference = engine.encode_ids(&[1, 2, 3], 3).ids;
        engine.set_profile(tiny_profile);
        let tiny = engine.encode_ids(&[1, 2, 3], 3).ids;
        engine.set_profile(indexed_profile);
        let indexed = engine.encode_ids(&[1, 2, 3], 3).ids;
        engine.set_profile(heap_profile);
        let heap = engine.encode_ids(&[1, 2, 3], 3).ids;

        assert_eq!(reference, tiny);
        assert_eq!(reference, indexed);
        assert_eq!(reference, heap);
    }
}
