//! Per-concept metadata and the deterministic retention-policy interface.
//!
//! All time is a caller-supplied **logical `u64` tick** — never wall-clock —
//! so the retention interface is bit-reproducible. Phase 1 defines the
//! [`RetentionPolicy`] trait and ships two deterministic policies
//! ([`NoForgetting`], [`LinearDecay`]) but runs **no** automatic eviction;
//! only explicit removal changes residency.

use crate::error::{HypermemoryError, Result};

/// Bookkeeping carried by every [`crate::ConceptRecord`].
///
/// Fields are private and mutated only through validating setters, so a stored
/// record can never acquire a non-finite importance. `insert_epoch` and
/// `sequence` are immutable after construction; the access fields
/// (`last_access`, `access_count`) and `active`/`importance` are mutable via the
/// store's guarded `get_mut`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConceptMetadata {
    importance: f32,
    insert_epoch: u64,
    last_access: u64,
    access_count: u64,
    active: bool,
}

impl ConceptMetadata {
    /// Construct metadata at insertion time `epoch` with the given importance.
    ///
    /// Returns [`HypermemoryError::InvalidRepresentation`] if `importance` is
    /// not finite and non-negative.
    pub(crate) fn new(importance: f32, epoch: u64) -> Result<Self> {
        Self::validate_importance(importance)?;
        Ok(Self {
            importance,
            insert_epoch: epoch,
            last_access: epoch,
            access_count: 0,
            active: true,
        })
    }

    fn validate_importance(importance: f32) -> Result<()> {
        if importance.is_finite() && importance >= 0.0
        {
            Ok(())
        }
        else
        {
            Err(HypermemoryError::InvalidRepresentation {
                reason: "importance must be finite and non-negative",
            })
        }
    }

    /// Importance weight (finite, non-negative).
    #[inline]
    #[must_use]
    pub const fn importance(&self) -> f32 {
        self.importance
    }

    /// Logical tick at which the concept was inserted.
    #[inline]
    #[must_use]
    pub const fn insert_epoch(&self) -> u64 {
        self.insert_epoch
    }

    /// Logical tick of the most recent access.
    #[inline]
    #[must_use]
    pub const fn last_access(&self) -> u64 {
        self.last_access
    }

    /// Number of times the concept has been accessed via the store's `touch`.
    #[inline]
    #[must_use]
    pub const fn access_count(&self) -> u64 {
        self.access_count
    }

    /// Whether the concept is active (a soft-state flag for retention
    /// experiments; residency is still governed by the store).
    #[inline]
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.active
    }

    /// Set importance, validating it is finite and non-negative.
    pub fn set_importance(&mut self, importance: f32) -> Result<()> {
        Self::validate_importance(importance)?;
        self.importance = importance;
        Ok(())
    }

    /// Set the active flag.
    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    /// Record an access at logical tick `now`: bumps `access_count`
    /// (saturating) and advances `last_access` monotonically (a `now` earlier
    /// than the current `last_access` never moves it backwards).
    pub(crate) fn record_access(&mut self, now: u64) {
        self.access_count = self.access_count.saturating_add(1);
        if now > self.last_access
        {
            self.last_access = now;
        }
    }
}

/// A deterministic retention score for a concept at logical tick `now`.
///
/// Higher means "more worth keeping". Phase 1 uses this only to *rank* residents
/// for later eviction experiments; it never evicts automatically. Implementors
/// must be pure and deterministic (no wall-clock, no RNG).
pub trait RetentionPolicy {
    /// Retention score at tick `now`. Must be finite for finite inputs.
    fn retention_score(&self, meta: &ConceptMetadata, now: u64) -> f32;
}

/// Keep everything: the retention score is exactly the importance. This is the
/// Phase 1 default — no recency term, no decay.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoForgetting;

impl RetentionPolicy for NoForgetting {
    #[inline]
    fn retention_score(&self, meta: &ConceptMetadata, _now: u64) -> f32 {
        meta.importance()
    }
}

/// Importance plus a deterministic integer-parameterized linear recency term.
///
/// `recency = max(0, 1 − age / half_life)` where `age = now − last_access`
/// (saturating; a `now` before `last_access` yields age 0, i.e. full recency).
/// `half_life` is a logical-tick span; `half_life == 0` disables the recency
/// term (treated as always-stale beyond the current tick). Fully deterministic:
/// integer age, one `f32` division.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearDecay {
    half_life: u64,
}

impl LinearDecay {
    /// New linear-decay policy with the given logical-tick half-life.
    #[must_use]
    pub const fn new(half_life: u64) -> Self {
        Self { half_life }
    }

    /// The configured half-life in logical ticks.
    #[inline]
    #[must_use]
    pub const fn half_life(&self) -> u64 {
        self.half_life
    }
}

impl RetentionPolicy for LinearDecay {
    fn retention_score(&self, meta: &ConceptMetadata, now: u64) -> f32 {
        let age = now.saturating_sub(meta.last_access());
        let recency = if self.half_life == 0
        {
            // No recency support: only the tick of last access itself counts.
            if age == 0 { 1.0 } else { 0.0 }
        }
        else
        {
            let r = 1.0 - (age as f32) / (self.half_life as f32);
            r.max(0.0)
        };
        meta.importance() + recency
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_non_finite_or_negative_importance() {
        assert!(ConceptMetadata::new(f32::NAN, 0).is_err());
        assert!(ConceptMetadata::new(-1.0, 0).is_err());
        assert!(ConceptMetadata::new(0.0, 0).is_ok());
        assert!(ConceptMetadata::new(2.5, 0).is_ok());
    }

    #[test]
    fn record_access_bumps_count_and_advances_monotonically() {
        let mut m = ConceptMetadata::new(1.0, 10).unwrap();
        assert_eq!(m.access_count(), 0);
        assert_eq!(m.last_access(), 10);
        m.record_access(20);
        assert_eq!(m.access_count(), 1);
        assert_eq!(m.last_access(), 20);
        // An earlier tick must not move last_access backwards.
        m.record_access(15);
        assert_eq!(m.access_count(), 2);
        assert_eq!(m.last_access(), 20);
    }

    #[test]
    fn set_importance_validates() {
        let mut m = ConceptMetadata::new(1.0, 0).unwrap();
        assert!(m.set_importance(f32::INFINITY).is_err());
        assert_eq!(m.importance(), 1.0);
        assert!(m.set_importance(3.0).is_ok());
        assert_eq!(m.importance(), 3.0);
    }

    #[test]
    fn no_forgetting_is_importance() {
        let m = ConceptMetadata::new(2.0, 0).unwrap();
        assert_eq!(NoForgetting.retention_score(&m, 1_000), 2.0);
    }

    #[test]
    fn linear_decay_is_deterministic_and_bounded() {
        let mut m = ConceptMetadata::new(1.0, 0).unwrap();
        m.record_access(100);
        let policy = LinearDecay::new(100);
        // age 0 → recency 1 → 2.0
        assert_eq!(policy.retention_score(&m, 100), 2.0);
        // age 50 → recency 0.5 → 1.5
        assert_eq!(policy.retention_score(&m, 150), 1.5);
        // age 200 → recency clamped to 0 → 1.0
        assert_eq!(policy.retention_score(&m, 300), 1.0);
        // Repeated evaluation is identical.
        assert_eq!(
            policy.retention_score(&m, 150),
            policy.retention_score(&m, 150)
        );
    }
}
