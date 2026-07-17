//! The stored concept: exact payload plus its hypercomplex representation.
//!
//! The sedenion is deliberately **not** the sole source of truth. Each record
//! keeps the exact byte payload, its content digest, the immutable `anchor`, the
//! (Phase 1: frozen) `residual`, the cached `effective` unit vector, the
//! insertion `sequence`, and mutable-but-guarded [`ConceptMetadata`].

use scirust_simd::hypercomplex::SedenionSimd;

use crate::digest::{Digest32, concept_digest};
use crate::error::Result;
use crate::id::ConceptId;
use crate::metadata::ConceptMetadata;
use crate::representation::effective_representation;

/// A single stored concept.
///
/// Construction is crate-internal ([`crate::S16Store::insert`]) so every record
/// is guaranteed to satisfy the insertion invariants: finite anchor and
/// residual, a computable unit `effective` vector, and a `content_digest` that
/// matches the `payload`. None of these can be violated after construction
/// because no public path exposes a raw `&mut ConceptRecord`.
#[derive(Clone, Debug, PartialEq)]
pub struct ConceptRecord {
    id: ConceptId,
    payload: Vec<u8>,
    content_digest: Digest32,
    anchor: SedenionSimd,
    residual: SedenionSimd,
    effective: SedenionSimd,
    metadata: ConceptMetadata,
    sequence: u64,
}

impl ConceptRecord {
    /// Build a validated record. Crate-internal.
    ///
    /// Computes and caches the effective representation once; because `anchor`
    /// and `residual` are immutable in Phase 1, the cache is valid for the
    /// record's entire lifetime. Fails if the effective representation is not
    /// computable (zero norm or non-finite).
    pub(crate) fn new(
        id: ConceptId,
        payload: Vec<u8>,
        anchor: SedenionSimd,
        residual: SedenionSimd,
        metadata: ConceptMetadata,
        sequence: u64,
    ) -> Result<Self> {
        let effective = effective_representation(&anchor, &residual)?;
        let content_digest = concept_digest(&payload);
        Ok(Self {
            id,
            payload,
            content_digest,
            anchor,
            residual,
            effective,
            metadata,
            sequence,
        })
    }

    /// The stable, generation-safe identifier.
    #[inline]
    #[must_use]
    pub const fn id(&self) -> ConceptId {
        self.id
    }

    /// The exact byte payload (the source of truth, not the sedenion).
    #[inline]
    #[must_use]
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    /// The domain-separated SHA-256 digest of the payload.
    #[inline]
    #[must_use]
    pub const fn content_digest(&self) -> &Digest32 {
        &self.content_digest
    }

    /// The immutable anchor representation. Relation composition uses this raw
    /// code (not the normalized [`Self::effective`] vector).
    #[inline]
    #[must_use]
    pub const fn anchor(&self) -> SedenionSimd {
        self.anchor
    }

    /// The (Phase 1: frozen) residual representation.
    #[inline]
    #[must_use]
    pub const fn residual(&self) -> SedenionSimd {
        self.residual
    }

    /// The cached effective representation `normalize(anchor + residual)`, a
    /// unit vector, used for similarity search.
    #[inline]
    #[must_use]
    pub const fn effective(&self) -> SedenionSimd {
        self.effective
    }

    /// The per-record metadata.
    #[inline]
    #[must_use]
    pub const fn metadata(&self) -> &ConceptMetadata {
        &self.metadata
    }

    /// The monotonic insertion sequence number (distinct from the logical
    /// insertion epoch in the metadata; this counts store insertions in order).
    #[inline]
    #[must_use]
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Guarded mutable access to the metadata, for the store's `get_mut` /
    /// `touch`. Not exposed publicly except through the store.
    #[inline]
    pub(crate) fn metadata_mut(&mut self) -> &mut ConceptMetadata {
        &mut self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> ConceptMetadata {
        ConceptMetadata::new(1.0, 0).unwrap()
    }

    #[test]
    fn record_caches_valid_effective_and_digest() {
        let anchor = SedenionSimd::unit(1) + SedenionSimd::unit(10);
        let rec = ConceptRecord::new(
            ConceptId::new(0, 0),
            b"payload".to_vec(),
            anchor,
            SedenionSimd::ZERO,
            meta(),
            0,
        )
        .unwrap();
        assert_eq!(rec.payload(), b"payload");
        assert_eq!(rec.content_digest(), &concept_digest(b"payload"));
        // effective is unit-norm.
        let n2 = crate::representation::norm_sqr_ordered(&rec.effective());
        assert!((n2 - 1.0).abs() < 1e-6);
        assert_eq!(rec.anchor().to_array(), anchor.to_array());
    }

    #[test]
    fn zero_anchor_without_residual_is_rejected() {
        let err = ConceptRecord::new(
            ConceptId::new(0, 0),
            b"x".to_vec(),
            SedenionSimd::ZERO,
            SedenionSimd::ZERO,
            meta(),
            0,
        );
        assert!(err.is_err());
    }
}
