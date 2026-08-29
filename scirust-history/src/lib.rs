//! Domain-neutral retained-history primitives for SciRust.
//!
//! This crate is a de-specialized harvest of history-management mechanisms
//! validated in `Memorithm/nonlocal-relativity-v2`. The source concepts are
//! principally the complete/bounded history work from research PR #1, the
//! true-position/non-uniform composition from PRs #6 and #7, the bounded
//! qualification experiment from PR #10, and the consolidated non-uniform
//! memory work from PR #13.
//!
//! The implementation here is adapted/reimplemented around generic `Value`
//! and `Position` types; it is not a verbatim move of the relativistic code.
//! General-relativistic coordinates, connections, transport, proper time and
//! memory laws deliberately remain outside this crate.
//!
//! # Semantic rules
//!
//! - Every retained value carries its original logical position explicitly.
//! - Accepted positions are strictly increasing. Duplicate, reversed, or
//!   incomparable positions fail closed and do not mutate history.
//! - [`CompleteHistory`] is the retained-history **reference** policy. The word
//!   reference does not imply an exact mathematical oracle for a downstream
//!   numerical method.
//! - [`BoundedHistory`] is always classified as an explicit approximation,
//!   even before its capacity has forced an eviction.
//! - Bounded eviction is observable through [`PushOutcome`]; truncation never
//!   occurs silently.
//! - No uniform-spacing assumption is made or reconstructed from an index.

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

extern crate alloc;

use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt;
use core::num::NonZeroUsize;

/// One retained historical value at its original logical position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HistoryEntry<Value, Position> {
    value: Value,
    position: Position,
}

impl<Value, Position> HistoryEntry<Value, Position> {
    /// Construct a history entry without altering or reconstructing its position.
    #[must_use]
    pub const fn new(value: Value, position: Position) -> Self {
        Self { value, position }
    }

    /// Return the retained value.
    #[must_use]
    pub const fn value(&self) -> &Value {
        &self.value
    }

    /// Return the original logical position supplied at insertion.
    #[must_use]
    pub const fn position(&self) -> &Position {
        &self.position
    }

    /// Decompose this entry into its retained value and original position.
    #[must_use]
    pub fn into_parts(self) -> (Value, Position) {
        (self.value, self.position)
    }
}

/// Scientific status of a retention policy relative to complete retained history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HistoryFidelity {
    /// Complete retained history used as the reference behavior.
    Reference,
    /// A deliberately reduced retained history that must be evaluated as an approximation.
    Approximation,
}

/// Storage-only retention policy for a history backend.
///
/// Product policy such as tenant quotas, attention semantics, authorization or
/// retrieval ranking does not belong in this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RetentionPolicy {
    /// Retain every accepted entry.
    Complete,
    /// Retain at most the given non-zero number of most recent entries.
    Bounded(NonZeroUsize),
}

impl RetentionPolicy {
    /// Construct a bounded policy, rejecting zero capacity explicitly.
    pub fn bounded(capacity: usize) -> Result<Self, HistoryError> {
        NonZeroUsize::new(capacity)
            .map(Self::Bounded)
            .ok_or(HistoryError::ZeroCapacity)
    }

    /// Return the scientific fidelity classification implied by this policy.
    #[must_use]
    pub const fn fidelity(self) -> HistoryFidelity {
        match self {
            Self::Complete => HistoryFidelity::Reference,
            Self::Bounded(_) => HistoryFidelity::Approximation,
        }
    }

    /// Return the maximum retained sample count, or `None` for complete retention.
    #[must_use]
    pub const fn capacity(self) -> Option<usize> {
        match self {
            Self::Complete => None,
            Self::Bounded(capacity) => Some(capacity.get()),
        }
    }
}

/// Failure to accept a history entry while preserving backend invariants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryError {
    /// A bounded backend was requested with zero retained capacity.
    ZeroCapacity,
    /// The proposed position is equal to the most recently accepted position.
    DuplicatePosition {
        /// Number of samples successfully accepted before the rejected proposal.
        observed_samples: usize,
    },
    /// The proposed position precedes the most recently accepted position.
    OutOfOrderPosition {
        /// Number of samples successfully accepted before the rejected proposal.
        observed_samples: usize,
    },
    /// The proposed position cannot be ordered with itself or the latest retained position.
    IncomparablePosition {
        /// Number of samples successfully accepted before the rejected proposal.
        observed_samples: usize,
    },
}

impl fmt::Display for HistoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCapacity => formatter.write_str("history retention capacity must be non-zero"),
            Self::DuplicatePosition { observed_samples } => write!(
                formatter,
                "history position duplicates the latest accepted position after {observed_samples} samples"
            ),
            Self::OutOfOrderPosition { observed_samples } => write!(
                formatter,
                "history position is older than the latest accepted position after {observed_samples} samples"
            ),
            Self::IncomparablePosition { observed_samples } => write!(
                formatter,
                "history position is not comparable after {observed_samples} samples"
            ),
        }
    }
}

/// Immutable view of the entries retained by one backend.
#[derive(Debug, Clone, Copy)]
pub struct HistoryView<'a, Value, Position> {
    entries: &'a [HistoryEntry<Value, Position>],
    policy: RetentionPolicy,
    observed_samples: usize,
}

impl<'a, Value, Position> HistoryView<'a, Value, Position> {
    const fn new(
        entries: &'a [HistoryEntry<Value, Position>],
        policy: RetentionPolicy,
        observed_samples: usize,
    ) -> Self {
        Self {
            entries,
            policy,
            observed_samples,
        }
    }

    /// Return retained entries from oldest retained position to newest.
    #[must_use]
    pub const fn entries(&self) -> &'a [HistoryEntry<Value, Position>] {
        self.entries
    }

    /// Return the number of entries currently retained.
    #[must_use]
    pub const fn retained_samples(&self) -> usize {
        self.entries.len()
    }

    /// Return the number of entries successfully accepted during this backend lifetime.
    #[must_use]
    pub const fn observed_samples(&self) -> usize {
        self.observed_samples
    }

    /// Return the configured storage-only retention policy.
    #[must_use]
    pub const fn retention_policy(&self) -> RetentionPolicy {
        self.policy
    }

    /// Return the reference-versus-approximation classification of the policy.
    #[must_use]
    pub const fn fidelity(&self) -> HistoryFidelity {
        self.policy.fidelity()
    }

    /// Return whether every successfully observed sample remains retained right now.
    ///
    /// This is distinct from [`Self::fidelity`]. A bounded policy remains an
    /// approximation policy even before its capacity has caused an eviction.
    #[must_use]
    pub const fn retains_all_observed(&self) -> bool {
        self.entries.len() == self.observed_samples
    }

    /// Return whether no entries are currently retained.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Result of one successful history insertion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PushOutcome<Value, Position> {
    evicted: Option<HistoryEntry<Value, Position>>,
    retained_samples: usize,
    observed_samples: usize,
}

impl<Value, Position> PushOutcome<Value, Position> {
    const fn new(
        evicted: Option<HistoryEntry<Value, Position>>,
        retained_samples: usize,
        observed_samples: usize,
    ) -> Self {
        Self {
            evicted,
            retained_samples,
            observed_samples,
        }
    }

    /// Return the entry evicted by this insertion, if bounded retention removed one.
    #[must_use]
    pub const fn evicted(&self) -> Option<&HistoryEntry<Value, Position>> {
        self.evicted.as_ref()
    }

    /// Consume the outcome and return the evicted entry, if any.
    #[must_use]
    pub fn into_evicted(self) -> Option<HistoryEntry<Value, Position>> {
        self.evicted
    }

    /// Return the retained sample count after the insertion.
    #[must_use]
    pub const fn retained_samples(&self) -> usize {
        self.retained_samples
    }

    /// Return the total successfully observed sample count after the insertion.
    #[must_use]
    pub const fn observed_samples(&self) -> usize {
        self.observed_samples
    }
}

/// Minimal contract implemented by deterministic history storage backends.
pub trait HistoryBackend<Value, Position>
where
    Position: PartialOrd,
{
    /// Return the storage-only retention policy.
    fn retention_policy(&self) -> RetentionPolicy;

    /// Return the total number of successfully accepted samples.
    fn observed_samples(&self) -> usize;

    /// Return an immutable ordered view of retained history.
    fn view(&self) -> HistoryView<'_, Value, Position>;

    /// Accept one entry while enforcing strict position ordering.
    ///
    /// Failed insertion must leave the backend unchanged. Any bounded eviction
    /// is returned explicitly in [`PushOutcome`].
    fn push(
        &mut self,
        entry: HistoryEntry<Value, Position>,
    ) -> Result<PushOutcome<Value, Position>, HistoryError>;
}

/// Complete retained-history reference backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompleteHistory<Value, Position> {
    entries: Vec<HistoryEntry<Value, Position>>,
}

impl<Value, Position> CompleteHistory<Value, Position> {
    /// Construct an empty complete-history backend.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Construct an empty complete-history backend with reserved capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }
}

impl<Value, Position> Default for CompleteHistory<Value, Position> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Value, Position> HistoryBackend<Value, Position> for CompleteHistory<Value, Position>
where
    Position: PartialOrd,
{
    fn retention_policy(&self) -> RetentionPolicy {
        RetentionPolicy::Complete
    }

    fn observed_samples(&self) -> usize {
        self.entries.len()
    }

    fn view(&self) -> HistoryView<'_, Value, Position> {
        HistoryView::new(&self.entries, RetentionPolicy::Complete, self.entries.len())
    }

    fn push(
        &mut self,
        entry: HistoryEntry<Value, Position>,
    ) -> Result<PushOutcome<Value, Position>, HistoryError> {
        let observed_samples = self.entries.len();
        validate_next_position(&self.entries, entry.position(), observed_samples)?;
        self.entries.push(entry);
        Ok(PushOutcome::new(None, self.entries.len(), self.entries.len()))
    }
}

/// Explicit bounded retained-history approximation backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundedHistory<Value, Position> {
    capacity: NonZeroUsize,
    entries: Vec<HistoryEntry<Value, Position>>,
    observed_samples: usize,
}

impl<Value, Position> BoundedHistory<Value, Position> {
    /// Construct an empty bounded-history backend.
    ///
    /// Capacity is a generic storage limit and may be one. Numerical kernels
    /// needing a larger stencil must enforce that requirement in their own layer.
    pub fn new(capacity: usize) -> Result<Self, HistoryError> {
        let capacity = NonZeroUsize::new(capacity).ok_or(HistoryError::ZeroCapacity)?;
        Ok(Self {
            capacity,
            entries: Vec::with_capacity(capacity.get()),
            observed_samples: 0,
        })
    }

    /// Return the maximum number of entries retained by this backend.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity.get()
    }
}

impl<Value, Position> HistoryBackend<Value, Position> for BoundedHistory<Value, Position>
where
    Position: PartialOrd,
{
    fn retention_policy(&self) -> RetentionPolicy {
        RetentionPolicy::Bounded(self.capacity)
    }

    fn observed_samples(&self) -> usize {
        self.observed_samples
    }

    fn view(&self) -> HistoryView<'_, Value, Position> {
        HistoryView::new(
            &self.entries,
            RetentionPolicy::Bounded(self.capacity),
            self.observed_samples,
        )
    }

    fn push(
        &mut self,
        entry: HistoryEntry<Value, Position>,
    ) -> Result<PushOutcome<Value, Position>, HistoryError> {
        validate_next_position(&self.entries, entry.position(), self.observed_samples)?;

        let evicted = if self.entries.len() == self.capacity.get() {
            Some(self.entries.remove(0))
        } else {
            None
        };
        self.entries.push(entry);
        self.observed_samples += 1;

        Ok(PushOutcome::new(
            evicted,
            self.entries.len(),
            self.observed_samples,
        ))
    }
}

fn validate_next_position<Value, Position>(
    entries: &[HistoryEntry<Value, Position>],
    position: &Position,
    observed_samples: usize,
) -> Result<(), HistoryError>
where
    Position: PartialOrd,
{
    if position.partial_cmp(position).is_none() {
        return Err(HistoryError::IncomparablePosition { observed_samples });
    }

    let Some(latest) = entries.last() else {
        return Ok(());
    };

    match latest.position().partial_cmp(position) {
        Some(Ordering::Less) => Ok(()),
        Some(Ordering::Equal) => Err(HistoryError::DuplicatePosition { observed_samples }),
        Some(Ordering::Greater) => Err(HistoryError::OutOfOrderPosition { observed_samples }),
        None => Err(HistoryError::IncomparablePosition { observed_samples }),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BoundedHistory, CompleteHistory, HistoryBackend, HistoryEntry, HistoryError,
        HistoryFidelity, RetentionPolicy,
    };

    #[test]
    fn complete_history_is_deterministic_reference() {
        let mut history = CompleteHistory::new();
        history.push(HistoryEntry::new(10_u64, 1_u64)).unwrap();
        history.push(HistoryEntry::new(20_u64, 3_u64)).unwrap();
        history.push(HistoryEntry::new(30_u64, 9_u64)).unwrap();

        let view = history.view();
        assert_eq!(view.fidelity(), HistoryFidelity::Reference);
        assert_eq!(view.retention_policy(), RetentionPolicy::Complete);
        assert_eq!(view.retained_samples(), 3);
        assert_eq!(view.observed_samples(), 3);
        assert!(view.retains_all_observed());
        assert_eq!(*view.entries()[0].position(), 1);
        assert_eq!(*view.entries()[1].position(), 3);
        assert_eq!(*view.entries()[2].position(), 9);
    }

    #[test]
    fn bounded_policy_is_explicit_approximation_before_eviction() {
        let mut history = BoundedHistory::new(4).unwrap();
        history.push(HistoryEntry::new(7_u8, 10_u64)).unwrap();

        let view = history.view();
        assert_eq!(view.fidelity(), HistoryFidelity::Approximation);
        assert_eq!(view.retention_policy().capacity(), Some(4));
        assert!(view.retains_all_observed());
    }

    #[test]
    fn bounded_eviction_is_observable_and_preserves_original_positions() {
        let mut history = BoundedHistory::new(2).unwrap();
        history.push(HistoryEntry::new('a', 1_u64)).unwrap();
        history.push(HistoryEntry::new('b', 8_u64)).unwrap();
        let outcome = history.push(HistoryEntry::new('c', 32_u64)).unwrap();

        let evicted = outcome.evicted().expect("third insertion must evict");
        assert_eq!(*evicted.value(), 'a');
        assert_eq!(*evicted.position(), 1);
        assert_eq!(outcome.retained_samples(), 2);
        assert_eq!(outcome.observed_samples(), 3);

        let positions: [u64; 2] = [
            *history.view().entries()[0].position(),
            *history.view().entries()[1].position(),
        ];
        assert_eq!(positions, [8, 32]);
        assert!(!history.view().retains_all_observed());
    }

    #[test]
    fn bounded_covering_all_samples_matches_complete_history_exactly() {
        let samples = [(2.5_f64, 0.0_f64), (-7.0, 0.125), (11.25, 0.9)];
        let mut complete = CompleteHistory::new();
        let mut bounded = BoundedHistory::new(samples.len()).unwrap();

        for (value, position) in samples {
            complete.push(HistoryEntry::new(value, position)).unwrap();
            let outcome = bounded.push(HistoryEntry::new(value, position)).unwrap();
            assert!(outcome.evicted().is_none());
        }

        assert_eq!(bounded.view().entries(), complete.view().entries());
        assert!(bounded.view().retains_all_observed());
        assert_eq!(bounded.view().fidelity(), HistoryFidelity::Approximation);
    }

    #[test]
    fn duplicate_position_is_rejected_without_mutation() {
        let mut history = CompleteHistory::new();
        history.push(HistoryEntry::new(1_u8, 4_u64)).unwrap();
        let before = history.clone();

        assert_eq!(
            history.push(HistoryEntry::new(2_u8, 4_u64)),
            Err(HistoryError::DuplicatePosition {
                observed_samples: 1
            })
        );
        assert_eq!(history, before);
    }

    #[test]
    fn reversed_position_is_rejected_without_mutation() {
        let mut history = BoundedHistory::new(2).unwrap();
        history.push(HistoryEntry::new(1_u8, 9_u64)).unwrap();
        let before = history.clone();

        assert_eq!(
            history.push(HistoryEntry::new(2_u8, 3_u64)),
            Err(HistoryError::OutOfOrderPosition {
                observed_samples: 1
            })
        );
        assert_eq!(history, before);
    }

    #[test]
    fn incomparable_position_is_rejected_without_mutation() {
        let mut history = CompleteHistory::<u8, f64>::new();
        assert_eq!(
            history.push(HistoryEntry::new(1, f64::NAN)),
            Err(HistoryError::IncomparablePosition {
                observed_samples: 0
            })
        );
        assert!(history.view().is_empty());
    }

    #[test]
    fn bounded_zero_capacity_is_rejected() {
        assert_eq!(
            BoundedHistory::<u8, u64>::new(0),
            Err(HistoryError::ZeroCapacity)
        );
        assert_eq!(RetentionPolicy::bounded(0), Err(HistoryError::ZeroCapacity));
    }

    #[test]
    fn bit_replay_is_deterministic_for_identical_insertions() {
        let samples = [
            (1.0_f64, 0.0_f64),
            (-0.0, 0.25),
            (f64::from_bits(0x3fe0_0000_0000_0001), 1.75),
        ];
        let mut left = CompleteHistory::new();
        let mut right = CompleteHistory::new();

        for (value, position) in samples {
            left.push(HistoryEntry::new(value, position)).unwrap();
            right.push(HistoryEntry::new(value, position)).unwrap();
        }

        for (left_entry, right_entry) in left
            .view()
            .entries()
            .iter()
            .zip(right.view().entries().iter())
        {
            assert_eq!(left_entry.value().to_bits(), right_entry.value().to_bits());
            assert_eq!(
                left_entry.position().to_bits(),
                right_entry.position().to_bits()
            );
        }
    }
}
