//! Generation-safe concept identifiers.
//!
//! A [`ConceptId`] pairs a slot index with the generation of the record that
//! occupied that slot when the id was issued. The store bumps a slot's
//! generation on removal, so a stale id can never resolve to the slot's next
//! occupant (see [`crate::S16Store`] invariant I1).

/// A stable, generation-safe handle to a concept.
///
/// The `Ord`/`PartialOrd` derives give a total order — lexicographic on
/// `(slot, generation)` — which is the deterministic tie-break used everywhere
/// ranking is involved. Fields are private; use [`ConceptId::slot`] and
/// [`ConceptId::generation`]. There is no public constructor: ids are minted
/// only by [`crate::S16Store`], so a caller cannot fabricate one that indexes
/// storage directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConceptId {
    slot: u32,
    generation: u32,
}

impl ConceptId {
    /// Construct an id. Crate-internal: only the store mints ids.
    #[inline]
    pub(crate) const fn new(slot: u32, generation: u32) -> Self {
        Self { slot, generation }
    }

    /// The slot index this id refers to.
    #[inline]
    #[must_use]
    pub const fn slot(self) -> u32 {
        self.slot
    }

    /// The generation this id was issued for.
    #[inline]
    #[must_use]
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accessors_round_trip() {
        let id = ConceptId::new(7, 3);
        assert_eq!(id.slot(), 7);
        assert_eq!(id.generation(), 3);
    }

    #[test]
    fn ordering_is_lexicographic_slot_then_generation() {
        // Deterministic tie-break: lower slot first, then lower generation.
        assert!(ConceptId::new(0, 9) < ConceptId::new(1, 0));
        assert!(ConceptId::new(1, 0) < ConceptId::new(1, 1));
        assert_eq!(ConceptId::new(2, 5), ConceptId::new(2, 5));
    }

    #[test]
    fn same_slot_different_generation_is_not_equal() {
        // The whole point of the generation field: two ids for the same reused
        // slot are distinct.
        assert_ne!(ConceptId::new(4, 0), ConceptId::new(4, 1));
    }
}
