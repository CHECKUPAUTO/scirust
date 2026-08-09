//! Contiguous compact merge-rule lookup for ElasticTokenizer hot paths.
//!
//! The compact representation stores sorted `(PairKey, PackedRule)` entries and
//! uses binary search. It is only constructed when every token ID and merge rank
//! fits the checked `u32` domain; callers keep their existing wide fallback.

use crate::elastic_id::{PackedRule, PairKey};
use crate::elastic_tokenizer::{DuplicateMergeRule, TokenId};

/// Sorted contiguous compact BPE rule table.
#[derive(Clone, Debug)]
pub(crate) struct FlatPackedRuleTable {
    entries: Vec<(PairKey, PackedRule)>,
}

impl FlatPackedRuleTable {
    /// Builds a compact table when every rule fits the `u32` packing domain.
    ///
    /// `Ok(None)` means the caller must use its wide compatibility table.
    pub(crate) fn try_from_ordered_merges(
        merges: &[(TokenId, TokenId, TokenId)],
    ) -> Result<Option<Self>, DuplicateMergeRule> {
        if !merges
            .iter()
            .enumerate()
            .all(|(rank, &(left, right, output))| {
                u32::try_from(rank).is_ok()
                    && u32::try_from(left).is_ok()
                    && u32::try_from(right).is_ok()
                    && u32::try_from(output).is_ok()
            })
        {
            return Ok(None);
        }

        let mut entries = Vec::with_capacity(merges.len());
        for (rank, &(left, right, output)) in merges.iter().enumerate()
        {
            let key = PairKey::try_from_usize(left, right)
                .expect("compact rule-table preflight checked token ids");
            let rule = PackedRule::try_from_usize(rank, output)
                .expect("compact rule-table preflight checked rule fields");
            entries.push((key, rule));
        }
        entries.sort_unstable_by_key(|(key, _)| *key);

        for duplicate in entries.windows(2)
        {
            if duplicate[0].0 == duplicate[1].0
            {
                let key = duplicate[0].0;
                return Err(DuplicateMergeRule {
                    left: usize::try_from(key.left()).expect("u32 token id fits usize"),
                    right: usize::try_from(key.right()).expect("u32 token id fits usize"),
                });
            }
        }

        Ok(Some(Self { entries }))
    }

    #[inline]
    pub(crate) fn get(&self, left: u32, right: u32) -> Option<PackedRule> {
        let key = PairKey::new(left, right);
        self.entries
            .binary_search_by_key(&key, |(entry_key, _)| *entry_key)
            .ok()
            .map(|index| self.entries[index].1)
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_is_exact_after_key_sorting() {
        let table =
            FlatPackedRuleTable::try_from_ordered_merges(&[(7, 9, 20), (1, 2, 21), (7, 3, 22)])
                .unwrap()
                .unwrap();
        assert_eq!(table.len(), 3);
        assert_eq!(table.get(7, 9).unwrap().rank(), 0);
        assert_eq!(table.get(7, 9).unwrap().output(), 20);
        assert_eq!(table.get(1, 2).unwrap().rank(), 1);
        assert_eq!(table.get(7, 3).unwrap().rank(), 2);
        assert_eq!(table.get(9, 7), None);
    }

    #[test]
    fn duplicate_pair_is_rejected_after_sorting() {
        let error =
            FlatPackedRuleTable::try_from_ordered_merges(&[(1, 2, 3), (1, 2, 4)]).unwrap_err();
        assert_eq!(error, DuplicateMergeRule { left: 1, right: 2 });
    }

    #[test]
    fn out_of_domain_rule_requests_wide_fallback() {
        if usize::BITS > 32
        {
            let wide = usize::try_from(u64::from(u32::MAX) + 1).unwrap();
            assert!(
                FlatPackedRuleTable::try_from_ordered_merges(&[(wide, 1, 2)])
                    .unwrap()
                    .is_none()
            );
        }
    }

    #[test]
    fn empty_table_is_valid_and_empty() {
        let table = FlatPackedRuleTable::try_from_ordered_merges(&[])
            .unwrap()
            .unwrap();
        assert!(table.is_empty());
    }
}
