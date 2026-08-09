//! Adaptive compact merge-rule lookup for ElasticTokenizer kernels.
//!
//! Normal canonical tokenizers use a left-indexed CSR table: `left_id` selects a
//! tiny contiguous `right_id` slice, then a binary search returns the packed rule.
//! If the offset vector would be disproportionately large, construction falls
//! back to a globally sorted flat packed table. Inputs outside the checked `u32`
//! token/rank domain return `Ok(None)` so the caller can keep its wide path.

use crate::elastic_id::{PackedRule, PairKey};
use crate::elastic_tokenizer::{DuplicateMergeRule, TokenId};

#[derive(Clone, Debug)]
pub(crate) enum AdaptivePackedRuleTable {
    Csr(CsrSoaRuleTable),
    Flat(FlatRuleTable),
}

#[derive(Clone, Debug)]
pub(crate) struct CsrSoaRuleTable {
    offsets: Vec<u32>,
    rights: Vec<u32>,
    rules: Vec<PackedRule>,
}

#[derive(Clone, Debug)]
pub(crate) struct FlatRuleTable {
    entries: Vec<(PairKey, PackedRule)>,
}

impl AdaptivePackedRuleTable {
    pub(crate) fn try_from_ordered_merges(
        merges: &[(TokenId, TokenId, TokenId)],
    ) -> Result<Option<Self>, DuplicateMergeRule> {
        let mut entries = Vec::with_capacity(merges.len());
        for (rank, &(left, right, output)) in merges.iter().enumerate()
        {
            let Ok(key) = PairKey::try_from_usize(left, right)
            else
            {
                return Ok(None);
            };
            let Ok(rule) = PackedRule::try_from_usize(rank, output)
            else
            {
                return Ok(None);
            };
            entries.push((key, rule));
        }
        entries.sort_unstable_by_key(|(key, _)| *key);
        reject_duplicates(&entries)?;

        if entries.is_empty()
        {
            return Ok(Some(Self::Flat(FlatRuleTable { entries })));
        }

        let max_left = entries
            .last()
            .map(|(key, _)| key.left())
            .expect("non-empty entries have a last element");
        let offset_len = usize::try_from(max_left)
            .ok()
            .and_then(|left| left.checked_add(2));
        let flat_payload_bytes = entries
            .len()
            .checked_mul(std::mem::size_of::<(PairKey, PackedRule)>());
        let rule_count_fits_u32 = u32::try_from(entries.len()).is_ok();

        // Bound CSR index memory by the payload size of the flat compact table.
        // Sparse/pathological high-left-ID tables therefore stay flat instead of
        // allocating an enormous mostly-empty offset vector. The total rule count
        // must also fit u32 so prefix offsets cannot overflow by construction.
        if let (Some(offset_len), Some(flat_payload_bytes)) = (offset_len, flat_payload_bytes)
        {
            let offset_bytes = offset_len.checked_mul(std::mem::size_of::<u32>());
            if rule_count_fits_u32
                && offset_bytes.is_some_and(|offsets| offsets <= flat_payload_bytes)
            {
                return Ok(Some(Self::Csr(CsrSoaRuleTable::from_sorted_entries(
                    entries, offset_len,
                ))));
            }
        }

        Ok(Some(Self::Flat(FlatRuleTable { entries })))
    }

    #[inline]
    pub(crate) fn get(&self, left: u32, right: u32) -> Option<PackedRule> {
        match self
        {
            Self::Csr(table) => table.get(left, right),
            Self::Flat(table) => table.get(left, right),
        }
    }

    #[cfg(test)]
    pub(crate) fn is_csr(&self) -> bool {
        matches!(self, Self::Csr(_))
    }
}

impl CsrSoaRuleTable {
    fn from_sorted_entries(entries: Vec<(PairKey, PackedRule)>, offset_len: usize) -> Self {
        let mut offsets = vec![0u32; offset_len];
        for &(key, _) in &entries
        {
            let left = usize::try_from(key.left()).expect("u32 left id fits usize");
            offsets[left + 1] = offsets[left + 1]
                .checked_add(1)
                .expect("bounded compact rule count fits u32");
        }
        for index in 1..offsets.len()
        {
            offsets[index] = offsets[index]
                .checked_add(offsets[index - 1])
                .expect("bounded compact rule count fits u32");
        }

        let mut rights = Vec::with_capacity(entries.len());
        let mut rules = Vec::with_capacity(entries.len());
        for (key, rule) in entries
        {
            rights.push(key.right());
            rules.push(rule);
        }
        Self {
            offsets,
            rights,
            rules,
        }
    }

    #[inline]
    fn get(&self, left: u32, right: u32) -> Option<PackedRule> {
        let left = usize::try_from(left).ok()?;
        if left + 1 >= self.offsets.len()
        {
            return None;
        }
        let start = usize::try_from(self.offsets[left]).ok()?;
        let end = usize::try_from(self.offsets[left + 1]).ok()?;
        self.rights[start..end]
            .binary_search(&right)
            .ok()
            .map(|local| self.rules[start + local])
    }
}

impl FlatRuleTable {
    #[inline]
    fn get(&self, left: u32, right: u32) -> Option<PackedRule> {
        let key = PairKey::new(left, right);
        self.entries
            .binary_search_by_key(&key, |(entry_key, _)| *entry_key)
            .ok()
            .map(|index| self.entries[index].1)
    }
}

fn reject_duplicates(entries: &[(PairKey, PackedRule)]) -> Result<(), DuplicateMergeRule> {
    for pair in entries.windows(2)
    {
        if pair[0].0 == pair[1].0
        {
            let key = pair[0].0;
            return Err(DuplicateMergeRule {
                left: usize::try_from(key.left()).expect("u32 token id fits usize"),
                right: usize::try_from(key.right()).expect("u32 token id fits usize"),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_rules_choose_csr_and_preserve_rank_payloads() {
        let merges = (0usize..512)
            .map(|left| (left, left + 1, left + 1000))
            .collect::<Vec<_>>();
        let table = AdaptivePackedRuleTable::try_from_ordered_merges(&merges)
            .unwrap()
            .unwrap();
        assert!(table.is_csr());
        for (rank, &(left, right, output)) in merges.iter().enumerate()
        {
            let rule = table
                .get(u32::try_from(left).unwrap(), u32::try_from(right).unwrap())
                .unwrap();
            assert_eq!(usize::try_from(rule.rank()).unwrap(), rank);
            assert_eq!(usize::try_from(rule.output()).unwrap(), output);
        }
    }

    #[test]
    fn sparse_high_left_id_uses_flat_fallback() {
        let table = AdaptivePackedRuleTable::try_from_ordered_merges(&[(100_000, 1, 2)])
            .unwrap()
            .unwrap();
        assert!(!table.is_csr());
        assert_eq!(table.get(100_000, 1).unwrap().output(), 2);
    }

    #[test]
    fn out_of_u32_domain_requests_wide_fallback() {
        if usize::BITS > 32
        {
            let wide = usize::try_from(u64::from(u32::MAX) + 1).unwrap();
            assert!(
                AdaptivePackedRuleTable::try_from_ordered_merges(&[(wide, 1, 2)])
                    .unwrap()
                    .is_none()
            );
        }
    }

    #[test]
    fn duplicate_pair_is_rejected() {
        let error = AdaptivePackedRuleTable::try_from_ordered_merges(&[(1, 2, 3), (1, 2, 4)])
            .unwrap_err();
        assert_eq!(error, DuplicateMergeRule { left: 1, right: 2 });
    }

    #[test]
    fn empty_table_is_valid() {
        let table = AdaptivePackedRuleTable::try_from_ordered_merges(&[])
            .unwrap()
            .unwrap();
        assert_eq!(table.get(1, 2), None);
    }
}
