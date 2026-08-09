//! Compact integer representations for ElasticTokenizer hot paths.
//!
//! Token IDs remain API-compatible for now. Internal hot-path fields can use
//! exact `u32` domains and pack two such values into one `u64` word without
//! changing tokenizer semantics.

use std::fmt;

/// `prev`/`next` sentinel representing no adjacent node.
pub const COMPACT_INDEX_NONE: u32 = u32::MAX;
/// `prev`/`next` sentinel representing a node removed by a merge.
pub const COMPACT_INDEX_INACTIVE: u32 = u32::MAX - 1;
/// Largest node index available without colliding with compact sentinels.
pub const COMPACT_INDEX_MAX: u32 = u32::MAX - 2;

#[inline]
pub fn try_compact_index(index: usize) -> Result<u32, CompactWordError> {
    let compact = u32::try_from(index).map_err(|_| CompactWordError::FieldTooWide {
        field: "node_index",
        value: index,
    })?;
    if compact > COMPACT_INDEX_MAX
    {
        return Err(CompactWordError::FieldTooWide {
            field: "node_index",
            value: index,
        });
    }
    Ok(compact)
}

/// Collision-free packed representation of an ordered `(left, right)` token pair.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PairKey(u64);

impl PairKey {
    #[inline]
    pub const fn new(left: u32, right: u32) -> Self {
        Self(((left as u64) << 32) | (right as u64))
    }

    #[inline]
    pub fn try_from_usize(left: usize, right: usize) -> Result<Self, PairKeyError> {
        let left = u32::try_from(left).map_err(|_| PairKeyError::TokenIdTooWide(left))?;
        let right = u32::try_from(right).map_err(|_| PairKeyError::TokenIdTooWide(right))?;
        Ok(Self::new(left, right))
    }

    #[inline]
    pub const fn left(self) -> u32 {
        high_u32(self.0)
    }

    #[inline]
    pub const fn right(self) -> u32 {
        low_u32(self.0)
    }

    #[inline]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Packed `(merge_rank, output_token_id)` rule payload.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PackedRule(u64);

impl PackedRule {
    #[inline]
    pub const fn new(rank: u32, output: u32) -> Self {
        Self(((rank as u64) << 32) | (output as u64))
    }

    #[inline]
    pub fn try_from_usize(rank: usize, output: usize) -> Result<Self, CompactWordError> {
        let rank = u32::try_from(rank).map_err(|_| CompactWordError::FieldTooWide {
            field: "merge_rank",
            value: rank,
        })?;
        let output = u32::try_from(output).map_err(|_| CompactWordError::FieldTooWide {
            field: "output_token_id",
            value: output,
        })?;
        Ok(Self::new(rank, output))
    }

    #[inline]
    pub const fn rank(self) -> u32 {
        high_u32(self.0)
    }

    #[inline]
    pub const fn output(self) -> u32 {
        low_u32(self.0)
    }

    #[inline]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Packed canonical scheduling key `(merge_rank, left_node_index)`.
///
/// Numeric `u64` ordering is exactly lexicographic ordering of these two `u32`
/// fields, so one integer comparison preserves rank priority and the left-most
/// tie break.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PriorityKey(u64);

impl PriorityKey {
    #[inline]
    pub const fn new(rank: u32, left_index: u32) -> Self {
        Self(((rank as u64) << 32) | (left_index as u64))
    }

    #[inline]
    pub fn try_from_usize(rank: usize, left_index: usize) -> Result<Self, CompactWordError> {
        let rank = u32::try_from(rank).map_err(|_| CompactWordError::FieldTooWide {
            field: "merge_rank",
            value: rank,
        })?;
        let left_index = try_compact_index(left_index)?;
        Ok(Self::new(rank, left_index))
    }

    #[inline]
    pub const fn rank(self) -> u32 {
        high_u32(self.0)
    }

    #[inline]
    pub const fn left_index(self) -> u32 {
        low_u32(self.0)
    }

    #[inline]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

const fn high_u32(word: u64) -> u32 {
    let bytes = word.to_be_bytes();
    u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

const fn low_u32(word: u64) -> u32 {
    let bytes = word.to_be_bytes();
    u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]])
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PairKeyError {
    TokenIdTooWide(usize),
}

impl fmt::Display for PairKeyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::TokenIdTooWide(value) =>
            {
                write!(f, "token id {value} exceeds the compact u32 domain")
            },
        }
    }
}

impl std::error::Error for PairKeyError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompactWordError {
    FieldTooWide { field: &'static str, value: usize },
}

impl fmt::Display for CompactWordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::FieldTooWide { field, value } =>
            {
                write!(f, "{field} value {value} exceeds the compact u32 domain")
            },
        }
    }
}

impl std::error::Error for CompactWordError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_words_are_exactly_one_u64() {
        assert_eq!(std::mem::size_of::<PairKey>(), 8);
        assert_eq!(std::mem::size_of::<PackedRule>(), 8);
        assert_eq!(std::mem::size_of::<PriorityKey>(), 8);
        if usize::BITS == 64
        {
            assert_eq!(std::mem::size_of::<(usize, usize)>(), 16);
        }
    }

    #[test]
    fn compact_index_sentinels_do_not_overlap_valid_indices() {
        assert!(COMPACT_INDEX_MAX < COMPACT_INDEX_INACTIVE);
        assert!(COMPACT_INDEX_INACTIVE < COMPACT_INDEX_NONE);
        assert_eq!(try_compact_index(0).unwrap(), 0);
        if usize::BITS > 32
        {
            let reserved = usize::try_from(COMPACT_INDEX_INACTIVE).unwrap();
            assert!(try_compact_index(reserved).is_err());
        }
    }

    #[test]
    fn pair_key_roundtrip_is_lossless() {
        for &(left, right) in &[
            (0, 0),
            (1, 2),
            (u32::MAX, 0),
            (0, u32::MAX),
            (u32::MAX, u32::MAX),
        ]
        {
            let key = PairKey::new(left, right);
            assert_eq!(key.left(), left);
            assert_eq!(key.right(), right);
        }
    }

    #[test]
    fn pair_key_order_matches_tuple_order() {
        let mut pairs = vec![(7u32, 9u32), (1, 8), (1, 2), (7, 3), (0, u32::MAX)];
        let mut keys = pairs
            .iter()
            .map(|&(left, right)| PairKey::new(left, right))
            .collect::<Vec<_>>();
        pairs.sort_unstable();
        keys.sort_unstable();
        let unpacked = keys
            .into_iter()
            .map(|key| (key.left(), key.right()))
            .collect::<Vec<_>>();
        assert_eq!(unpacked, pairs);
    }

    #[test]
    fn packed_rule_roundtrip_is_lossless() {
        let rule = PackedRule::new(123, 456);
        assert_eq!(rule.rank(), 123);
        assert_eq!(rule.output(), 456);
    }

    #[test]
    fn priority_key_order_matches_semantic_tuple_order() {
        let mut priorities = vec![(5u32, 2u32), (1, 90), (1, 3), (9, 0), (5, 1)];
        let mut keys = priorities
            .iter()
            .map(|&(rank, left)| PriorityKey::new(rank, left))
            .collect::<Vec<_>>();
        priorities.sort_unstable();
        keys.sort_unstable();
        let unpacked = keys
            .into_iter()
            .map(|key| (key.rank(), key.left_index()))
            .collect::<Vec<_>>();
        assert_eq!(unpacked, priorities);
    }

    #[test]
    fn usize_narrowing_is_checked() {
        assert_eq!(PairKey::try_from_usize(4, 5).unwrap(), PairKey::new(4, 5));
        assert_eq!(
            PackedRule::try_from_usize(6, 7).unwrap(),
            PackedRule::new(6, 7)
        );
        assert_eq!(
            PriorityKey::try_from_usize(8, 9).unwrap(),
            PriorityKey::new(8, 9)
        );
        if usize::BITS > 32
        {
            let too_wide = usize::try_from(u64::from(u32::MAX) + 1).unwrap();
            assert!(matches!(
                PairKey::try_from_usize(too_wide, 0),
                Err(PairKeyError::TokenIdTooWide(value)) if value == too_wide
            ));
            assert!(PackedRule::try_from_usize(too_wide, 0).is_err());
            assert!(PriorityKey::try_from_usize(0, too_wide).is_err());
        }
    }
}
