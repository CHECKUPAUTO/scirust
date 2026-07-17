//! The crate's single, non-panicking public error type.
//!
//! No public runtime path in `scirust-hypermemory` uses `unwrap`, `expect`, or
//! indexing that can panic; every recoverable failure is one of these variants.
//! The enum is `#[non_exhaustive]` because this is a research crate that will
//! grow new failure modes in later phases — external code must keep a wildcard
//! arm.

use core::fmt;

/// Every recoverable failure the public API can produce.
///
/// Variants are grouped by the required distinctions in the Phase 1 spec:
/// representation validity, identifier validity, relation/expression validity,
/// capacity, and internal invariants.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum HypermemoryError {
    /// A representation was structurally invalid for a reason other than the
    /// more specific zero-norm / non-finite cases (e.g. a residual exceeding its
    /// configured bound, or an out-of-range importance).
    InvalidRepresentation {
        /// Human-readable, stable reason string (not localized).
        reason: &'static str,
    },
    /// A representation whose effective vector has zero norm, so it cannot be
    /// normalized or indexed.
    ZeroNormRepresentation,
    /// A representation containing a non-finite lane (NaN / ±∞), or whose norm
    /// overflowed to a non-finite value.
    NonFiniteRepresentation,
    /// A residual representation exceeded the store's configured bound.
    ResidualOutOfBounds {
        /// The configured maximum residual norm.
        bound: f32,
        /// The offending residual norm.
        got: f32,
    },
    /// A [`crate::ConceptId`] referenced a slot that is occupied by a *different*
    /// generation — the original concept was removed (and possibly the slot
    /// reused). Stale ids never resolve to the new occupant.
    StaleId {
        /// The slot index the id pointed at.
        slot: u32,
        /// The generation the id carried.
        id_generation: u32,
        /// The generation currently occupying the slot.
        current_generation: u32,
    },
    /// A [`crate::ConceptId`] referenced a slot that currently holds no record
    /// (removed or never inserted).
    VacantSlot {
        /// The slot index the id pointed at.
        slot: u32,
    },
    /// A [`crate::ConceptId`] referenced a slot index beyond the store's slot
    /// array.
    SlotOutOfRange {
        /// The slot index the id pointed at.
        slot: u32,
        /// The number of slots that exist.
        slots: u32,
    },
    /// The identifier space is exhausted: no free slot exists and a new slot
    /// cannot be appended without exceeding `u32::MAX`.
    IdSpaceExhausted,
    /// The store's optional capacity is full.
    CapacityExhausted {
        /// The configured capacity.
        capacity: usize,
    },
    /// A relation atom referenced a concept that is not present (vacant slot or
    /// out-of-range). Distinguished from [`Self::StaleId`], which is a
    /// generation mismatch.
    MissingAtom {
        /// The slot index the atom referenced.
        slot: u32,
    },
    /// A relation was structurally invalid (e.g. an internal inconsistency
    /// detected while walking a caller-constructed expression tree).
    InvalidRelation {
        /// Human-readable, stable reason string.
        reason: &'static str,
    },
    /// An expression exceeded the configured maximum depth.
    ExpressionDepthLimit {
        /// The configured maximum depth.
        limit: usize,
    },
    /// An expression exceeded the configured maximum node count.
    ExpressionSizeLimit {
        /// The configured maximum size (node count).
        limit: usize,
    },
    /// A recoverable internal invariant violation. Reaching this is a bug, but
    /// it is surfaced as an error rather than a panic so a caller can recover.
    InvariantViolation {
        /// Human-readable, stable detail string.
        detail: &'static str,
    },
}

impl fmt::Display for HypermemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidRepresentation { reason } =>
            {
                write!(f, "invalid representation: {reason}")
            },
            Self::ZeroNormRepresentation =>
            {
                write!(f, "representation has zero norm and cannot be normalized")
            },
            Self::NonFiniteRepresentation =>
            {
                write!(
                    f,
                    "representation contains a non-finite (NaN or infinite) value"
                )
            },
            Self::ResidualOutOfBounds { bound, got } =>
            {
                write!(f, "residual norm {got} exceeds configured bound {bound}")
            },
            Self::StaleId {
                slot,
                id_generation,
                current_generation,
            } =>
            {
                write!(
                    f,
                    "stale ConceptId: slot {slot} generation {id_generation} \
                     (current generation is {current_generation})"
                )
            },
            Self::VacantSlot { slot } => write!(f, "slot {slot} is vacant"),
            Self::SlotOutOfRange { slot, slots } =>
            {
                write!(f, "slot {slot} is out of range (only {slots} slots exist)")
            },
            Self::IdSpaceExhausted =>
            {
                write!(
                    f,
                    "identifier space exhausted (no free slot and u32 slot index full)"
                )
            },
            Self::CapacityExhausted { capacity } =>
            {
                write!(f, "store capacity {capacity} exhausted")
            },
            Self::MissingAtom { slot } =>
            {
                write!(f, "relation atom references missing concept at slot {slot}")
            },
            Self::InvalidRelation { reason } => write!(f, "invalid relation: {reason}"),
            Self::ExpressionDepthLimit { limit } =>
            {
                write!(f, "expression exceeds maximum depth {limit}")
            },
            Self::ExpressionSizeLimit { limit } =>
            {
                write!(f, "expression exceeds maximum size {limit}")
            },
            Self::InvariantViolation { detail } =>
            {
                write!(f, "internal invariant violation: {detail}")
            },
        }
    }
}

impl std::error::Error for HypermemoryError {}

/// Convenience alias for results in this crate.
pub type Result<T> = core::result::Result<T, HypermemoryError>;
