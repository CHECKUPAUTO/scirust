//! Compact integer representations for ElasticTokenizer hot paths.
//!
//! Token IDs remain API-compatible for now. This module starts the internal
//! migration by packing two `u32` token IDs into one exact `u64` merge key.
//! No hashing shortcut or lossy narrowing is involved.

use std::fmt;

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
        let bytes = self.0.to_be_bytes();
        u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    #[inline]
    pub const fn right(self) -> u32 {
        let bytes = self.0.to_be_bytes();
        u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]])
    }

    #[inline]
    pub const fn raw(self) -> u64 {
        self.0
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_key_is_exactly_one_u64() {
        assert_eq!(std::mem::size_of::<PairKey>(), 8);
        if usize::BITS == 64
        {
            assert_eq!(std::mem::size_of::<(usize, usize)>(), 16);
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
    fn usize_narrowing_is_checked() {
        assert_eq!(PairKey::try_from_usize(4, 5).unwrap(), PairKey::new(4, 5));
        if usize::BITS > 32
        {
            let too_wide = usize::try_from(u64::from(u32::MAX) + 1).unwrap();
            assert!(matches!(
                PairKey::try_from_usize(too_wide, 0),
                Err(PairKeyError::TokenIdTooWide(value)) if value == too_wide
            ));
        }
    }
}
