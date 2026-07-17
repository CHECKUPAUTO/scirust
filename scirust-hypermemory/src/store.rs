//! The deterministic slot/generation concept store.
//!
//! `S16Store` is a `Vec<Slot>` plus an explicit free list. It provides
//! generation-safe insertion, lookup, guarded mutation, removal with slot
//! reuse, and strictly slot-index-ordered (i.e. deterministic) iteration. There
//! is no `HashMap` anywhere in the observable API, and no public path hands out
//! a raw `&mut ConceptRecord` — mutation is confined to the metadata.
//!
//! See `docs/research/SCIRUST_HYPERMEMORY_PHASE1.md` §7 for invariants I1–I6.

use scirust_simd::hypercomplex::SedenionSimd;

use crate::error::{HypermemoryError, Result};
use crate::id::ConceptId;
use crate::metadata::ConceptMetadata;
use crate::record::ConceptRecord;

/// The default upper bound on a residual's norm. Phase 1 defaults residuals to
/// zero, so this only ever gates an explicitly supplied residual.
pub const DEFAULT_RESIDUAL_BOUND: f32 = 1.0;

/// Everything needed to insert one concept.
///
/// Build with [`ConceptSpec::new`] (residual defaults to the zero sedenion —
/// Phase 1 keeps learning disabled) and optionally [`ConceptSpec::with_residual`].
#[derive(Clone, Debug, PartialEq)]
pub struct ConceptSpec {
    /// The exact byte payload (the source of truth).
    pub payload: Vec<u8>,
    /// The immutable anchor representation.
    pub anchor: SedenionSimd,
    /// The residual representation (default: zero).
    pub residual: SedenionSimd,
    /// Importance weight (must be finite and non-negative).
    pub importance: f32,
    /// Logical insertion tick (never wall-clock).
    pub tick: u64,
}

impl ConceptSpec {
    /// A spec with a zero residual (the Phase 1 default).
    #[must_use]
    pub fn new(payload: Vec<u8>, anchor: SedenionSimd, importance: f32, tick: u64) -> Self {
        Self {
            payload,
            anchor,
            residual: SedenionSimd::ZERO,
            importance,
            tick,
        }
    }

    /// Attach an explicit residual (subject to the store's residual bound).
    #[must_use]
    pub fn with_residual(mut self, residual: SedenionSimd) -> Self {
        self.residual = residual;
        self
    }
}

/// One physical slot in the store.
#[derive(Clone, Debug, PartialEq)]
enum Slot {
    /// Holds a live record; `generation` matches the record's id generation.
    Occupied {
        generation: u32,
        record: Box<ConceptRecord>,
    },
    /// Empty and available for reuse at the given (already-incremented)
    /// generation.
    Vacant { generation: u32 },
    /// Permanently retired: its generation counter would have overflowed
    /// `u32::MAX`, so reusing it could collide with a previously issued id. It
    /// is never handed back out.
    Retired,
}

/// A deterministic, generation-safe store of [`ConceptRecord`]s.
#[derive(Clone, Debug)]
pub struct S16Store {
    slots: Vec<Slot>,
    free: Vec<u32>,
    len: usize,
    next_sequence: u64,
    residual_bound: f32,
    capacity: Option<usize>,
}

impl Default for S16Store {
    fn default() -> Self {
        Self::new()
    }
}

impl S16Store {
    /// An unbounded store with the default residual bound.
    #[must_use]
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            len: 0,
            next_sequence: 0,
            residual_bound: DEFAULT_RESIDUAL_BOUND,
            capacity: None,
        }
    }

    /// A store with a fixed maximum number of live concepts.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity: Some(capacity),
            ..Self::new()
        }
    }

    /// Set the residual-norm bound (used to validate explicitly supplied
    /// residuals). Must be finite and non-negative; otherwise the previous bound
    /// is kept and an error is returned.
    pub fn set_residual_bound(&mut self, bound: f32) -> Result<()> {
        if bound.is_finite() && bound >= 0.0
        {
            self.residual_bound = bound;
            Ok(())
        }
        else
        {
            Err(HypermemoryError::InvalidRepresentation {
                reason: "residual bound must be finite and non-negative",
            })
        }
    }

    /// The configured residual-norm bound.
    #[inline]
    #[must_use]
    pub const fn residual_bound(&self) -> f32 {
        self.residual_bound
    }

    /// Number of live concepts.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the store holds no live concepts.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The optional capacity, if one was configured.
    #[inline]
    #[must_use]
    pub const fn capacity(&self) -> Option<usize> {
        self.capacity
    }

    /// Insert a concept, returning its fresh generation-safe id.
    ///
    /// Validates (invariant I6): finite anchor and residual, residual within
    /// bound, finite non-negative importance, and a computable effective
    /// representation. On any failure nothing is stored and the error is
    /// returned. Reuses a free slot (incrementing its generation) before
    /// appending a new one.
    pub fn insert(&mut self, spec: ConceptSpec) -> Result<ConceptId> {
        // Capacity check first — cheap, and avoids doing validation work we would
        // only discard.
        if let Some(cap) = self.capacity
        {
            if self.len >= cap
            {
                return Err(HypermemoryError::CapacityExhausted { capacity: cap });
            }
        }

        // Validate the residual bound (finite already implied by a finite norm;
        // an explicit check gives the precise error).
        crate::representation::validate_finite(&spec.residual)?;
        let residual_norm_sqr = crate::representation::norm_sqr_ordered(&spec.residual);
        if !residual_norm_sqr.is_finite()
        {
            return Err(HypermemoryError::NonFiniteRepresentation);
        }
        let bound_sqr = self.residual_bound * self.residual_bound;
        if residual_norm_sqr > bound_sqr
        {
            return Err(HypermemoryError::ResidualOutOfBounds {
                bound: self.residual_bound,
                got: residual_norm_sqr.sqrt(),
            });
        }

        let metadata = ConceptMetadata::new(spec.importance, spec.tick)?;
        let sequence = self.next_sequence;

        // Decide the slot & generation, but do not commit until the record is
        // built successfully (so a validation failure leaves the store pristine).
        let (slot_index, generation, reuse) = match self.free.pop()
        {
            Some(idx) =>
            {
                let generation = match self.slots.get(idx as usize)
                {
                    Some(Slot::Vacant { generation }) => *generation,
                    _ =>
                    {
                        // The free list and slots disagree — recoverable bug.
                        return Err(HypermemoryError::InvariantViolation {
                            detail: "free-list entry did not point at a vacant slot",
                        });
                    },
                };
                (idx, generation, true)
            },
            None =>
            {
                let idx = self.slots.len();
                if idx > u32::MAX as usize
                {
                    return Err(HypermemoryError::IdSpaceExhausted);
                }
                (idx as u32, 0u32, false)
            },
        };

        let id = ConceptId::new(slot_index, generation);
        let record = ConceptRecord::new(
            id,
            spec.payload,
            spec.anchor,
            spec.residual,
            metadata,
            sequence,
        )?;

        // Commit.
        let occupied = Slot::Occupied {
            generation,
            record: Box::new(record),
        };
        if reuse
        {
            self.slots[slot_index as usize] = occupied;
        }
        else
        {
            self.slots.push(occupied);
        }
        self.len += 1;
        self.next_sequence += 1;
        Ok(id)
    }

    /// Resolve an id to a slot index, distinguishing every failure mode.
    fn resolve(&self, id: ConceptId) -> Result<usize> {
        let idx = id.slot() as usize;
        match self.slots.get(idx)
        {
            None => Err(HypermemoryError::SlotOutOfRange {
                slot: id.slot(),
                slots: self.slots.len() as u32,
            }),
            Some(Slot::Vacant { .. }) | Some(Slot::Retired) =>
            {
                Err(HypermemoryError::VacantSlot { slot: id.slot() })
            },
            Some(Slot::Occupied { generation, .. }) =>
            {
                if *generation == id.generation()
                {
                    Ok(idx)
                }
                else
                {
                    Err(HypermemoryError::StaleId {
                        slot: id.slot(),
                        id_generation: id.generation(),
                        current_generation: *generation,
                    })
                }
            },
        }
    }

    /// Borrow the record for `id`, or the precise error if it does not resolve.
    pub fn get(&self, id: ConceptId) -> Result<&ConceptRecord> {
        let idx = self.resolve(id)?;
        match &self.slots[idx]
        {
            Slot::Occupied { record, .. } => Ok(record),
            _ => Err(HypermemoryError::InvariantViolation {
                detail: "resolved slot was not occupied",
            }),
        }
    }

    /// Guarded mutable access to a concept's metadata (invariant I5).
    ///
    /// Only the metadata is mutable — the payload, anchor, residual, cached
    /// effective vector, and content digest are never exposed mutably, so the
    /// digest and effective representation stay consistent for the record's
    /// whole lifetime. Named `get_mut` to match the store API surface; the
    /// returned `&mut ConceptMetadata` can only be mutated through its own
    /// validating methods (its fields are private).
    pub fn get_mut(&mut self, id: ConceptId) -> Result<&mut ConceptMetadata> {
        let idx = self.resolve(id)?;
        match &mut self.slots[idx]
        {
            Slot::Occupied { record, .. } => Ok(record.metadata_mut()),
            _ => Err(HypermemoryError::InvariantViolation {
                detail: "resolved slot was not occupied",
            }),
        }
    }

    /// Record an access at logical tick `now` (bumps access count, advances
    /// last-access monotonically).
    pub fn touch(&mut self, id: ConceptId, now: u64) -> Result<()> {
        self.get_mut(id)?.record_access(now);
        Ok(())
    }

    /// Whether `id` currently resolves to a live record.
    #[must_use]
    pub fn contains(&self, id: ConceptId) -> bool {
        self.resolve(id).is_ok()
    }

    /// Remove the concept `id`, freeing its slot for generation-safe reuse.
    ///
    /// Bumps the slot's generation so every previously issued id for this slot
    /// becomes stale (invariant I1). If the generation would overflow the slot
    /// is retired and never reused (invariant I2). Returns the removed record.
    pub fn remove(&mut self, id: ConceptId) -> Result<ConceptRecord> {
        let idx = self.resolve(id)?;
        // `resolve` guarantees Occupied with a matching generation.
        let slot = &mut self.slots[idx];
        let (generation, record) = match core::mem::replace(slot, Slot::Retired)
        {
            Slot::Occupied { generation, record } => (generation, record),
            other =>
            {
                // Restore and report — must not lose the slot's state.
                *slot = other;
                return Err(HypermemoryError::InvariantViolation {
                    detail: "removed slot was not occupied",
                });
            },
        };

        match next_generation(generation)
        {
            Some(next) =>
            {
                *slot = Slot::Vacant { generation: next };
                self.free.push(idx as u32);
            },
            None =>
            {
                // Overflow: leave it Retired (already set above), do not free.
            },
        }
        self.len -= 1;
        Ok(*record)
    }

    /// Deterministic iterator over live records in ascending slot-index order.
    pub fn iter(&self) -> impl Iterator<Item = &ConceptRecord> + '_ {
        self.slots.iter().filter_map(|slot| match slot
        {
            Slot::Occupied { record, .. } => Some(record.as_ref()),
            _ => None,
        })
    }

    /// Deterministic iterator over live ids in ascending slot-index order.
    pub fn ids(&self) -> impl Iterator<Item = ConceptId> + '_ {
        self.iter().map(ConceptRecord::id)
    }

    /// Test-only: force a slot's generation, to exercise overflow retirement
    /// without performing four billion removals.
    #[cfg(test)]
    pub(crate) fn force_slot_generation(&mut self, id: ConceptId, generation: u32) -> ConceptId {
        let idx = id.slot() as usize;
        if let Slot::Occupied {
            generation: g,
            record,
        } = &mut self.slots[idx]
        {
            *g = generation;
            let new_id = ConceptId::new(id.slot(), generation);
            // Keep the record's own id consistent so `get` still matches.
            let payload = record.payload().to_vec();
            let anchor = record.anchor();
            let residual = record.residual();
            let meta = *record.metadata();
            let seq = record.sequence();
            **record = ConceptRecord::new(new_id, payload, anchor, residual, meta, seq)
                .expect("test helper rebuilds a record from already-valid parts");
            new_id
        }
        else
        {
            id
        }
    }
}

/// The next generation for a slot on removal, or `None` on `u32::MAX` overflow.
///
/// Factored out so the overflow boundary is unit-testable directly.
#[inline]
const fn next_generation(generation: u32) -> Option<u32> {
    generation.checked_add(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(tag: &[u8], unit: usize) -> ConceptSpec {
        ConceptSpec::new(tag.to_vec(), SedenionSimd::unit(unit), 1.0, 0)
    }

    #[test]
    fn first_insertion_resolves() {
        let mut store = S16Store::new();
        let id = store.insert(spec(b"a", 0)).unwrap();
        assert_eq!(id.slot(), 0);
        assert_eq!(id.generation(), 0);
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
        assert_eq!(store.get(id).unwrap().payload(), b"a");
        assert!(store.contains(id));
    }

    #[test]
    fn removal_frees_and_reports_gone() {
        let mut store = S16Store::new();
        let id = store.insert(spec(b"a", 0)).unwrap();
        let rec = store.remove(id).unwrap();
        assert_eq!(rec.payload(), b"a");
        assert_eq!(store.len(), 0);
        assert!(!store.contains(id));
    }

    #[test]
    fn removed_slot_reports_vacant() {
        let mut store = S16Store::new();
        let id = store.insert(spec(b"a", 0)).unwrap();
        store.remove(id).unwrap();
        assert_eq!(store.get(id), Err(HypermemoryError::VacantSlot { slot: 0 }));
    }

    #[test]
    fn stale_id_is_rejected_after_slot_reuse() {
        let mut store = S16Store::new();
        let old = store.insert(spec(b"a", 0)).unwrap();
        store.remove(old).unwrap();
        // Reuse the freed slot: same index, generation incremented.
        let new = store.insert(spec(b"b", 1)).unwrap();
        assert_eq!(new.slot(), old.slot());
        assert_eq!(new.generation(), old.generation() + 1);
        // The stale id must never resolve to the new occupant.
        assert_eq!(
            store.get(old),
            Err(HypermemoryError::StaleId {
                slot: 0,
                id_generation: 0,
                current_generation: 1,
            })
        );
        assert_eq!(store.get(new).unwrap().payload(), b"b");
    }

    #[test]
    fn slot_reuse_prefers_free_list_before_appending() {
        let mut store = S16Store::new();
        let a = store.insert(spec(b"a", 0)).unwrap();
        let b = store.insert(spec(b"b", 1)).unwrap();
        assert_eq!((a.slot(), b.slot()), (0, 1));
        store.remove(a).unwrap();
        // Next insert reuses slot 0 (the only free slot), not slot 2.
        let c = store.insert(spec(b"c", 2)).unwrap();
        assert_eq!(c.slot(), 0);
        assert_eq!(c.generation(), 1);
    }

    #[test]
    fn deterministic_iteration_in_slot_order() {
        let mut store = S16Store::new();
        let a = store.insert(spec(b"a", 0)).unwrap();
        let b = store.insert(spec(b"b", 1)).unwrap();
        let c = store.insert(spec(b"c", 2)).unwrap();
        store.remove(b).unwrap();
        // Live: slot 0 (a), slot 2 (c) — b's slot 1 is vacant.
        let ids: Vec<_> = store.ids().collect();
        assert_eq!(ids, vec![a, c]);
        // Reinsert reuses slot 1; iteration is still slot-ordered a, d, c.
        let d = store.insert(spec(b"d", 3)).unwrap();
        assert_eq!(d.slot(), 1);
        let payloads: Vec<_> = store.iter().map(|r| r.payload().to_vec()).collect();
        assert_eq!(payloads, vec![b"a".to_vec(), b"d".to_vec(), b"c".to_vec()]);
    }

    #[test]
    fn capacity_boundary_is_enforced() {
        let mut store = S16Store::with_capacity(2);
        let _a = store.insert(spec(b"a", 0)).unwrap();
        let b = store.insert(spec(b"b", 1)).unwrap();
        assert_eq!(
            store.insert(spec(b"c", 2)),
            Err(HypermemoryError::CapacityExhausted { capacity: 2 })
        );
        // Freeing a slot makes room again.
        store.remove(b).unwrap();
        assert!(store.insert(spec(b"c", 2)).is_ok());
    }

    #[test]
    fn out_of_range_id_is_rejected() {
        let store = S16Store::new();
        let bogus = ConceptId::new(5, 0);
        assert_eq!(
            store.get(bogus),
            Err(HypermemoryError::SlotOutOfRange { slot: 5, slots: 0 })
        );
    }

    #[test]
    fn next_generation_overflow_boundary() {
        assert_eq!(next_generation(0), Some(1));
        assert_eq!(next_generation(u32::MAX - 1), Some(u32::MAX));
        assert_eq!(next_generation(u32::MAX), None);
    }

    #[test]
    fn generation_overflow_retires_slot_without_reuse() {
        let mut store = S16Store::new();
        let id = store.insert(spec(b"a", 0)).unwrap();
        // Age the slot to the last usable generation.
        let aged = store.force_slot_generation(id, u32::MAX);
        assert_eq!(store.get(aged).unwrap().payload(), b"a");
        // Removing at u32::MAX overflows → slot retired, not freed.
        store.remove(aged).unwrap();
        assert_eq!(store.len(), 0);
        // The retired slot is never reused: the next insert appends a new slot.
        let next = store.insert(spec(b"b", 1)).unwrap();
        assert_eq!(next.slot(), 1, "retired slot 0 must not be reused");
        // The retired slot reports vacant, never resolves.
        assert_eq!(
            store.get(aged),
            Err(HypermemoryError::VacantSlot { slot: 0 })
        );
    }

    #[test]
    fn get_mut_touch_updates_access_metadata() {
        let mut store = S16Store::new();
        let id = store.insert(spec(b"a", 0)).unwrap();
        store.touch(id, 42).unwrap();
        let m = store.get(id).unwrap().metadata();
        assert_eq!(m.access_count(), 1);
        assert_eq!(m.last_access(), 42);
        store.get_mut(id).unwrap().set_importance(9.0).unwrap();
        assert_eq!(store.get(id).unwrap().metadata().importance(), 9.0);
    }

    #[test]
    fn residual_bound_is_enforced() {
        let mut store = S16Store::new();
        store.set_residual_bound(0.5).unwrap();
        // Residual norm 1.0 > bound 0.5 → rejected.
        let big_residual = SedenionSimd::unit(3);
        let spec = ConceptSpec::new(b"x".to_vec(), SedenionSimd::unit(0), 1.0, 0)
            .with_residual(big_residual);
        match store.insert(spec)
        {
            Err(HypermemoryError::ResidualOutOfBounds { bound, got }) =>
            {
                assert_eq!(bound, 0.5);
                assert!((got - 1.0).abs() < 1e-6);
            },
            other => panic!("expected ResidualOutOfBounds, got {other:?}"),
        }
        // A residual within bound is accepted.
        let ok = ConceptSpec::new(b"y".to_vec(), SedenionSimd::unit(0), 1.0, 0)
            .with_residual(SedenionSimd::unit(3).scale(0.25));
        assert!(store.insert(ok).is_ok());
    }

    #[test]
    fn invalid_representation_is_not_stored() {
        let mut store = S16Store::new();
        // Zero anchor + zero residual → no effective vector.
        let bad = ConceptSpec::new(b"z".to_vec(), SedenionSimd::ZERO, 1.0, 0);
        assert!(store.insert(bad).is_err());
        assert_eq!(store.len(), 0, "a rejected insert must not grow the store");
    }
}
