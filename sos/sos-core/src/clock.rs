//! [`LamportClock`] — the authoritative logical time on every object.
//!
//! Wall-clock timestamps are neither portable nor reproducible (and would make
//! object hashes machine-dependent), so SOS makes a **logical** Lamport clock
//! the authoritative happens-before order and keeps wall-clock only as advisory
//! side-car metadata excluded from the hash (RFC-0002 §03, SDE RFC §03.5).

use serde::{Deserialize, Serialize};

use crate::canonical::{Canonical, CanonicalEncoder};

/// A scalar Lamport logical clock.
///
/// Ordering is by the underlying counter; equal counters are concurrent and are
/// broken deterministically elsewhere (by [`crate::ObjectId`]).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub struct LamportClock(pub u64);

impl LamportClock {
    /// The zero clock — the logical time of a root object with no history.
    pub const ZERO: Self = Self(0);

    /// The clock one tick after this one (a local event / a derived object).
    #[must_use]
    pub const fn tick(self) -> Self {
        Self(self.0 + 1)
    }

    /// The clock after observing `other`: `max(self, other) + 1`.
    ///
    /// This is the Lamport receive rule — used when an object is derived from
    /// several parents, so its logical time strictly follows all of them.
    #[must_use]
    pub fn observe(self, other: Self) -> Self {
        Self(self.0.max(other.0) + 1)
    }

    /// The raw counter value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl Canonical for LamportClock {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.u64(self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_advances_by_one() {
        assert_eq!(LamportClock::ZERO.tick(), LamportClock(1));
        assert_eq!(LamportClock(41).tick(), LamportClock(42));
    }

    #[test]
    fn observe_follows_all_parents() {
        // Deriving from parents at logical times 3 and 7 must land at 8.
        let derived = LamportClock(3).observe(LamportClock(7));
        assert_eq!(derived, LamportClock(8));
        assert!(derived > LamportClock(7));
        assert!(derived > LamportClock(3));
    }

    #[test]
    fn ordering_is_by_counter() {
        assert!(LamportClock(1) < LamportClock(2));
        assert_eq!(LamportClock::ZERO, LamportClock(0));
    }
}
