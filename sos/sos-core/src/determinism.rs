//! The determinism taxonomy — honest, four-level, propagated by `meet`.
//!
//! A reproducibility claim is worthless as a bare boolean. Every SOS object
//! declares the [`DeterminismLevel`] it *realized*, and the level propagates as
//! the **minimum** (`meet`) along any dependency path: one stochastic ancestor
//! makes the whole downstream chain stochastic, and the object graph says so
//! rather than a README claiming "reproducible". See RFC-0002 §09 and the SDE
//! RFC §01.6.

use serde::{Deserialize, Serialize};

use crate::canonical::{Canonical, CanonicalEncoder};

/// How reproducible an object is, from strongest (`L3`) to weakest (`L0`).
///
/// The numeric order is meaningful: `L0 < L1 < L2 < L3`, so the weakest level
/// along a path is `min` / [`DeterminismLevel::meet`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DeterminismLevel {
    /// **L0 — non-deterministic, but recorded.** Not reproducible in principle
    /// (a physical measurement, a live feed, a human), but the observation is
    /// recorded, so downstream replay is `L3` against the recording.
    L0,
    /// **L1 — statistically reproducible.** Identical in distribution given the
    /// recorded seed and sampler (Monte-Carlo, stochastic optimization).
    L1,
    /// **L2 — numerically reproducible.** Identical up to a declared tolerance,
    /// with a certificate bounding the deviation (cross-hardware BLAS, GPU
    /// reductions, iterative solvers to a tolerance).
    L2,
    /// **L3 — bit-reproducible.** Byte-identical output for identical input on
    /// any conforming machine (integer/symbolic code, fixed-order arithmetic).
    L3,
}

impl DeterminismLevel {
    /// The **weakest** of two levels — the propagation operator. `meet` is
    /// commutative, associative, and idempotent, with identity [`Self::L3`].
    #[must_use]
    pub fn meet(self, other: Self) -> Self {
        self.min(other)
    }

    /// The weakest level over an iterator of levels, i.e. the realized level of
    /// an object given the levels of everything it depends on.
    ///
    /// An **empty** iterator yields [`Self::L3`]: a thing that depends on
    /// nothing stochastic is, so far, bit-reproducible. `L3` is the identity of
    /// `meet`, which is exactly why the empty fold returns it.
    #[must_use]
    pub fn min_over<I: IntoIterator<Item = Self>>(levels: I) -> Self {
        levels.into_iter().fold(Self::L3, Self::meet)
    }

    /// A short, stable code (`"L0"`..`"L3"`) — matches
    /// `scirust-bench-schema::Certificate.determinism` (`D0`..`D3` maps
    /// one-to-one) so SOS and SciRust share one vocabulary.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self
        {
            Self::L0 => "L0",
            Self::L1 => "L1",
            Self::L2 => "L2",
            Self::L3 => "L3",
        }
    }
}

impl Canonical for DeterminismLevel {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        // Encode as the ordinal so it is compact and order-preserving.
        enc.u64(*self as u64);
    }
}

#[cfg(test)]
mod tests {
    use super::DeterminismLevel::{L0, L1, L2, L3};
    use super::*;

    #[test]
    fn ordering_is_strongest_last() {
        assert!(L0 < L1 && L1 < L2 && L2 < L3);
    }

    #[test]
    fn meet_takes_the_weakest() {
        assert_eq!(L3.meet(L1), L1);
        assert_eq!(L0.meet(L3), L0);
        assert_eq!(L2.meet(L2), L2);
    }

    #[test]
    fn meet_is_commutative_and_idempotent() {
        for a in [L0, L1, L2, L3]
        {
            assert_eq!(a.meet(a), a);
            for b in [L0, L1, L2, L3]
            {
                assert_eq!(a.meet(b), b.meet(a));
            }
        }
    }

    #[test]
    fn min_over_empty_is_l3_identity() {
        assert_eq!(DeterminismLevel::min_over(std::iter::empty()), L3);
    }

    #[test]
    fn min_over_propagates_weakest() {
        assert_eq!(DeterminismLevel::min_over([L3, L3, L1, L3]), L1);
        assert_eq!(DeterminismLevel::min_over([L3, L2]), L2);
        assert_eq!(DeterminismLevel::min_over([L0, L1, L2, L3]), L0);
    }

    #[test]
    fn codes_and_serde_are_stable() {
        assert_eq!(L2.code(), "L2");
        assert_eq!(serde_json::to_string(&L2).unwrap(), "\"L2\"");
        let back: DeterminismLevel = serde_json::from_str("\"L0\"").unwrap();
        assert_eq!(back, L0);
    }
}
