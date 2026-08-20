//! Representation metadata for elastic KV-cache materializations.
//!
//! This module is intentionally runtime-neutral.  It gives SciRust's KV-cache
//! implementations a deterministic vocabulary for describing *what* is stored
//! without depending on ElasticXxx, EPG, FLAT-ATTENTION, or a particular model.
//! Higher-level runtimes may map these values into their own policy/transition
//! graphs.

use core::fmt;

/// Monotonic version of one materialized KV representation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KvRepresentationEpoch(u64);

impl KvRepresentationEpoch {
    /// Construct an epoch from its raw counter.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Return the raw epoch counter.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Return the checked successor epoch.
    pub fn next(self) -> Result<Self, KvRepresentationError> {
        self.0
            .checked_add(1)
            .map(Self)
            .ok_or(KvRepresentationError::EpochOverflow)
    }
}

/// Stable, runtime-neutral representation contract identifier.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KvRepresentationId(String);

impl KvRepresentationId {
    /// Construct a non-empty identifier such as `raw.f32`, `slha.int4-r1`, or
    /// `epg.so4.structural`.
    pub fn new(value: impl Into<String>) -> Result<Self, KvRepresentationError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(KvRepresentationError::EmptyRepresentationId);
        }
        Ok(Self(value))
    }

    /// Borrow the identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Relationship between a stored key and positional/structural transforms.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum KvKeyTransformScope {
    /// Key is stored before positional/structural transformation.
    #[default]
    Raw,
    /// Key is transformed only from token/page-stable metadata. This proves
    /// that the transform itself is independent of a future query; it does not
    /// by itself prove cross-query reuse compatibility.
    TokenStable,
    /// Key materialization depends on a particular query or future context.
    QueryDependent,
}

impl KvKeyTransformScope {
    /// Whether the stored key transform is independent of a future query.
    ///
    /// This is a necessary but not sufficient condition for cross-query KV
    /// reuse. Derivation provenance, model/schema compatibility, positions,
    /// epochs, and other domain contracts may still make two materializations
    /// incompatible.
    pub const fn is_query_independent(self) -> bool {
        !matches!(self, Self::QueryDependent)
    }
}

/// Metadata attached to a materialized KV tile/page.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvRepresentationMetadata {
    /// Stable representation family identifier.
    pub id: KvRepresentationId,
    /// Schema/contract version for the representation family.
    pub schema_version: u32,
    /// Epoch of the current materialization.
    pub epoch: KvRepresentationEpoch,
    /// Positional/structural transform scope of the stored key.
    pub key_transform_scope: KvKeyTransformScope,
}

impl KvRepresentationMetadata {
    /// Construct metadata.
    pub const fn new(
        id: KvRepresentationId,
        schema_version: u32,
        epoch: KvRepresentationEpoch,
        key_transform_scope: KvKeyTransformScope,
    ) -> Self {
        Self {
            id,
            schema_version,
            epoch,
            key_transform_scope,
        }
    }

    /// Whether this metadata's key transform is independent of a future query.
    ///
    /// This local predicate does not establish complete reuse compatibility.
    pub const fn key_transform_is_query_independent(&self) -> bool {
        self.key_transform_scope.is_query_independent()
    }

    /// Whether another metadata value names the same representation contract,
    /// ignoring the materialization epoch.
    pub fn same_contract(&self, rhs: &Self) -> bool {
        self.id == rhs.id && self.schema_version == rhs.schema_version
    }

    /// Validate the epoch relationship for replacing this materialization with
    /// `target`.
    ///
    /// Every committed replacement materialization must advance the epoch,
    /// including a re-encode/recompute that keeps the same representation
    /// contract. Contract changes retain a dedicated diagnostic because they
    /// must never be silently committed at the current epoch.
    pub fn validate_successor(&self, target: &Self) -> Result<(), KvRepresentationError> {
        if target.epoch < self.epoch {
            return Err(KvRepresentationError::EpochRegression {
                from: self.epoch,
                to: target.epoch,
            });
        }
        if target.epoch == self.epoch {
            if self.same_contract(target) {
                return Err(KvRepresentationError::MaterializationMustAdvanceEpoch {
                    from: self.epoch,
                    to: target.epoch,
                });
            }
            return Err(KvRepresentationError::ContractChangeMustAdvanceEpoch {
                from: self.epoch,
                to: target.epoch,
            });
        }
        Ok(())
    }
}

/// Ordered semantics used to materialize a stored key.
///
/// The order is represented explicitly because a positional/structural
/// transform and a numerical codec or quantizer do not generally commute. A
/// concrete application may establish an equivalence for a particular pair of
/// operations, but this generic metadata does not assume one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum KvKeyEncodingOrder {
    /// K remains in its raw transform domain.
    #[default]
    Raw,
    /// Apply the key transform before the numerical codec/quantizer.
    TransformThenCodec,
    /// Apply the numerical codec/domain conversion before the key transform.
    CodecThenTransform,
    /// A fused implementation with semantics named by the representation
    /// contract; no equivalence with either ordered form is implied.
    FusedDeclared,
}

/// Descriptive source retained for a future KV rematerialization.
///
/// This records scientific/runtime metadata only. It is not an authorization
/// token and does not prove that a future re-encode or recomputation is safe.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum KvRematerializationSource {
    /// No independent source is declared.
    #[default]
    None,
    /// A canonical/raw K source is retained independently of this materialized
    /// representation.
    StoredCanonicalRaw,
    /// K may be regenerated from an upstream model state.
    ModelRecompute,
    /// A stable application-defined external source is retained.
    ExternalStableSource,
}

/// Additive materialization descriptor combining the existing representation
/// metadata with transform/codec ordering and rematerialization provenance.
///
/// This type is deliberately separate from [`KvRepresentationMetadata`] so the
/// existing SciRust metadata constructor and represented-cache API remain
/// source-compatible.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvMaterializationDescriptor {
    /// Existing representation identity, epoch and transform scope.
    pub representation: KvRepresentationMetadata,
    /// Exact ordered semantics used for K materialization.
    pub key_encoding_order: KvKeyEncodingOrder,
    /// Descriptive source retained for a later rematerialization.
    pub rematerialization_source: KvRematerializationSource,
}

impl KvMaterializationDescriptor {
    /// Construct and validate a materialization descriptor.
    pub fn new(
        representation: KvRepresentationMetadata,
        key_encoding_order: KvKeyEncodingOrder,
        rematerialization_source: KvRematerializationSource,
    ) -> Result<Self, KvRepresentationError> {
        let descriptor = Self {
            representation,
            key_encoding_order,
            rematerialization_source,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    /// Validate the local transform-scope/encoding-order relationship.
    pub fn validate(&self) -> Result<(), KvRepresentationError> {
        match (
            self.representation.key_transform_scope,
            self.key_encoding_order,
        ) {
            (KvKeyTransformScope::Raw, KvKeyEncodingOrder::Raw) => Ok(()),
            (KvKeyTransformScope::Raw, _) => {
                Err(KvRepresentationError::RawKeyHasTransformEncoding)
            }
            (_, KvKeyEncodingOrder::Raw) => {
                Err(KvRepresentationError::TransformedKeyMissingEncoding)
            }
            _ => Ok(()),
        }
    }

    /// Whether the key transform itself is independent of future queries.
    ///
    /// As on [`KvRepresentationMetadata`], this is necessary but not sufficient
    /// for complete cross-query cache reuse.
    pub const fn key_transform_is_query_independent(&self) -> bool {
        self.representation
            .key_transform_is_query_independent()
    }

    /// Validate a replacement descriptor.
    ///
    /// Both descriptors must be locally valid and every replacement must obey
    /// the existing representation epoch-successor rule. This method describes
    /// metadata consistency only; it does not perform or authorize the physical
    /// transformation.
    pub fn validate_successor(&self, target: &Self) -> Result<(), KvRepresentationError> {
        self.validate()?;
        target.validate()?;
        self.representation
            .validate_successor(&target.representation)
    }
}

/// Invalid representation metadata or epoch transition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KvRepresentationError {
    /// Representation identifiers may not be blank.
    EmptyRepresentationId,
    /// Epoch counter cannot be incremented further.
    EpochOverflow,
    /// A raw key declared a transform/codec encoding order.
    RawKeyHasTransformEncoding,
    /// A transformed key failed to declare a non-raw encoding order.
    TransformedKeyMissingEncoding,
    /// A contract-changing materialization did not advance the epoch.
    ContractChangeMustAdvanceEpoch {
        /// Source epoch.
        from: KvRepresentationEpoch,
        /// Target epoch.
        to: KvRepresentationEpoch,
    },
    /// A replacement materialization using the same contract did not advance
    /// the epoch.
    MaterializationMustAdvanceEpoch {
        /// Source epoch.
        from: KvRepresentationEpoch,
        /// Target epoch.
        to: KvRepresentationEpoch,
    },
    /// A materialization regressed its epoch.
    EpochRegression {
        /// Source epoch.
        from: KvRepresentationEpoch,
        /// Target epoch.
        to: KvRepresentationEpoch,
    },
}

impl fmt::Display for KvRepresentationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyRepresentationId => {
                write!(f, "KV representation identifier must not be empty")
            }
            Self::EpochOverflow => write!(f, "KV representation epoch overflow"),
            Self::RawKeyHasTransformEncoding => {
                write!(f, "raw K must use the raw key-encoding order")
            }
            Self::TransformedKeyMissingEncoding => {
                write!(f, "transformed K must declare a non-raw key-encoding order")
            }
            Self::ContractChangeMustAdvanceEpoch { from, to } => write!(
                f,
                "changing KV representation contract must advance epoch ({} -> {})",
                from.get(),
                to.get()
            ),
            Self::MaterializationMustAdvanceEpoch { from, to } => write!(
                f,
                "replacing KV materialization must advance epoch ({} -> {})",
                from.get(),
                to.get()
            ),
            Self::EpochRegression { from, to } => write!(
                f,
                "KV representation epoch regressed ({} -> {})",
                from.get(),
                to.get()
            ),
        }
    }
}

impl std::error::Error for KvRepresentationError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(name: &str, epoch: u64, scope: KvKeyTransformScope) -> KvRepresentationMetadata {
        KvRepresentationMetadata::new(
            KvRepresentationId::new(name).unwrap(),
            1,
            KvRepresentationEpoch::new(epoch),
            scope,
        )
    }

    #[test]
    fn query_independence_is_only_a_local_transform_property() {
        let query_dependent = meta("epg.dynamic", 2, KvKeyTransformScope::QueryDependent);
        let token_stable = meta("epg.so4", 2, KvKeyTransformScope::TokenStable);
        assert!(!query_dependent.key_transform_is_query_independent());
        assert!(token_stable.key_transform_is_query_independent());
    }

    #[test]
    fn contract_change_requires_new_epoch() {
        let from = meta("epg.so2", 7, KvKeyTransformScope::TokenStable);
        let bad = meta("epg.so4", 7, KvKeyTransformScope::TokenStable);
        let good = meta("epg.so4", 8, KvKeyTransformScope::TokenStable);
        assert!(matches!(
            from.validate_successor(&bad),
            Err(KvRepresentationError::ContractChangeMustAdvanceEpoch { .. })
        ));
        assert!(from.validate_successor(&good).is_ok());
    }

    #[test]
    fn same_contract_replacement_requires_new_epoch() {
        let from = meta("kv.int4", 4, KvKeyTransformScope::TokenStable);
        let bad = meta("kv.int4", 4, KvKeyTransformScope::TokenStable);
        let good = meta("kv.int4", 5, KvKeyTransformScope::TokenStable);
        assert!(matches!(
            from.validate_successor(&bad),
            Err(KvRepresentationError::MaterializationMustAdvanceEpoch { .. })
        ));
        assert!(from.validate_successor(&good).is_ok());
    }

    #[test]
    fn successor_rejects_epoch_regression_before_contract_diagnostics() {
        let from = meta("kv.int4", 4, KvKeyTransformScope::TokenStable);
        let target = meta("kv.int8", 3, KvKeyTransformScope::TokenStable);
        assert!(matches!(
            from.validate_successor(&target),
            Err(KvRepresentationError::EpochRegression { .. })
        ));
    }

    #[test]
    fn materialization_descriptor_requires_explicit_transform_order() {
        let raw = meta("raw.f16", 1, KvKeyTransformScope::Raw);
        assert!(KvMaterializationDescriptor::new(
            raw.clone(),
            KvKeyEncodingOrder::Raw,
            KvRematerializationSource::None,
        )
        .is_ok());
        assert_eq!(
            KvMaterializationDescriptor::new(
                raw,
                KvKeyEncodingOrder::TransformThenCodec,
                KvRematerializationSource::None,
            ),
            Err(KvRepresentationError::RawKeyHasTransformEncoding)
        );

        let transformed = meta("epg.so4", 2, KvKeyTransformScope::TokenStable);
        assert_eq!(
            KvMaterializationDescriptor::new(
                transformed,
                KvKeyEncodingOrder::Raw,
                KvRematerializationSource::StoredCanonicalRaw,
            ),
            Err(KvRepresentationError::TransformedKeyMissingEncoding)
        );
    }

    #[test]
    fn materialization_descriptor_preserves_non_commuting_order_identity() {
        let transformed = meta("epg.so4.int4", 2, KvKeyTransformScope::TokenStable);
        let transform_then_codec = KvMaterializationDescriptor::new(
            transformed.clone(),
            KvKeyEncodingOrder::TransformThenCodec,
            KvRematerializationSource::StoredCanonicalRaw,
        )
        .unwrap();
        let codec_then_transform = KvMaterializationDescriptor::new(
            transformed,
            KvKeyEncodingOrder::CodecThenTransform,
            KvRematerializationSource::StoredCanonicalRaw,
        )
        .unwrap();
        assert_ne!(transform_then_codec, codec_then_transform);
    }

    #[test]
    fn materialization_successor_keeps_existing_epoch_rule() {
        let from = KvMaterializationDescriptor::new(
            meta("epg.so4.int4", 9, KvKeyTransformScope::TokenStable),
            KvKeyEncodingOrder::TransformThenCodec,
            KvRematerializationSource::StoredCanonicalRaw,
        )
        .unwrap();
        let same_epoch = KvMaterializationDescriptor::new(
            meta("epg.so4.int4", 9, KvKeyTransformScope::TokenStable),
            KvKeyEncodingOrder::CodecThenTransform,
            KvRematerializationSource::StoredCanonicalRaw,
        )
        .unwrap();
        let next_epoch = KvMaterializationDescriptor::new(
            meta("epg.so4.int4", 10, KvKeyTransformScope::TokenStable),
            KvKeyEncodingOrder::CodecThenTransform,
            KvRematerializationSource::StoredCanonicalRaw,
        )
        .unwrap();

        assert!(matches!(
            from.validate_successor(&same_epoch),
            Err(KvRepresentationError::MaterializationMustAdvanceEpoch { .. })
        ));
        assert!(from.validate_successor(&next_epoch).is_ok());
    }
}
