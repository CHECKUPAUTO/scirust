//! Exact physical-segment accounting for tensor representations.
//!
//! This module is intentionally policy-free. It describes owned physical
//! segments, references to shared segments, and exact serialized/resident bit
//! totals. It does not choose representations, quantizers, sparse formats, or
//! allocation policies.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::fmt;

use crate::representation::StorageBits;

/// Stable identifier of one physical segment inside an accounting scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PhysicalSegmentId(u32);

impl PhysicalSegmentId {
    /// Construct an identifier from its canonical integer value.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Return the canonical integer value.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Stable identity of the bytes/bitstream contents required by a segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContentIdentity(u64);

impl ContentIdentity {
    /// Construct an opaque content identity.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Return the opaque identity value.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Stable identity of the physical encoding/layout of a segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LayoutIdentity(u64);

impl LayoutIdentity {
    /// Construct an opaque layout identity.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Return the opaque identity value.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Named class of resident materialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaterializationClass(u32);

impl MaterializationClass {
    /// Construct a materialization-class identifier.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Return the canonical identifier value.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Semantic role of one physically accounted segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum PhysicalSegmentRole {
    /// Main encoded payload.
    Payload,
    /// Sparse/indexing payload.
    Index,
    /// Scale or dequantization parameter payload.
    Scale,
    /// Shared dictionary/codebook payload.
    Codebook,
    /// Format/header metadata.
    Metadata,
    /// Auxiliary residual payload.
    Residual,
    /// Other explicitly accounted auxiliary state.
    Auxiliary,
}

/// Lifetime at which physical ownership is meaningful.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum SegmentLifetime {
    /// Shared for a model/checkpoint lifetime.
    ModelStatic,
    /// Shared for a graph/compiled-program lifetime.
    GraphStatic,
    /// Owned for one request.
    Request,
    /// Owned for one sequence/session.
    Sequence,
    /// Owned for one tensor block/tile.
    Block,
}

/// Reconstruction semantics relevant to zero-bit physical storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum ReconstructionRole {
    /// Stored physical data participates in reconstruction.
    Stored,
    /// No physical bits are needed because the logical value is explicitly the
    /// constant zero value under the enclosing representation contract.
    ImplicitZero,
}

/// Exact resident extent for one named materialization class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentMaterialization {
    class: MaterializationClass,
    bits: StorageBits,
    alignment_bits: u64,
}

impl ResidentMaterialization {
    /// Construct one exact resident materialization description.
    pub fn new(
        class: MaterializationClass,
        bits: StorageBits,
        alignment_bits: u64,
    ) -> Result<Self, PhysicalAccountingError> {
        validate_alignment(alignment_bits)?;
        Ok(Self {
            class,
            bits,
            alignment_bits,
        })
    }

    /// Materialization class.
    pub const fn class(&self) -> MaterializationClass {
        self.class
    }

    /// Exact resident extent in physical bits.
    pub const fn bits(&self) -> StorageBits {
        self.bits
    }

    /// Required resident alignment in bits.
    pub const fn alignment_bits(&self) -> u64 {
        self.alignment_bits
    }
}

/// One owned physical segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalSegment {
    id: PhysicalSegmentId,
    role: PhysicalSegmentRole,
    content_identity: ContentIdentity,
    layout_identity: LayoutIdentity,
    raw_bits: StorageBits,
    serialized_bits: StorageBits,
    serialized_alignment_bits: u64,
    lifetime: SegmentLifetime,
    reconstruction_role: ReconstructionRole,
    resident_materializations: Vec<ResidentMaterialization>,
}

impl PhysicalSegment {
    /// Construct an owned physical segment.
    ///
    /// `raw_bits` is the exact encoded extent before required container padding;
    /// `serialized_bits` includes that padding. Resident extents are supplied
    /// separately because they are materialization-specific.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: PhysicalSegmentId,
        role: PhysicalSegmentRole,
        content_identity: ContentIdentity,
        layout_identity: LayoutIdentity,
        raw_bits: StorageBits,
        serialized_bits: StorageBits,
        serialized_alignment_bits: u64,
        lifetime: SegmentLifetime,
        reconstruction_role: ReconstructionRole,
        resident_materializations: Vec<ResidentMaterialization>,
    ) -> Result<Self, PhysicalAccountingError> {
        validate_alignment(serialized_alignment_bits)?;
        if serialized_bits.get() < raw_bits.get() {
            return Err(PhysicalAccountingError::SerializedBelowRaw { id });
        }
        if serialized_bits.get() == 0 && reconstruction_role != ReconstructionRole::ImplicitZero {
            return Err(PhysicalAccountingError::ZeroBitWithoutReconstruction { id });
        }
        if reconstruction_role == ReconstructionRole::ImplicitZero
            && (raw_bits.get() != 0 || serialized_bits.get() != 0)
        {
            return Err(PhysicalAccountingError::ImplicitZeroHasStorage { id });
        }

        let mut seen = BTreeMap::new();
        for materialization in &resident_materializations {
            if materialization.bits.get() < raw_bits.get() {
                return Err(PhysicalAccountingError::ResidentBelowRaw {
                    id,
                    class: materialization.class,
                });
            }
            if seen.insert(materialization.class, ()).is_some() {
                return Err(PhysicalAccountingError::DuplicateMaterialization {
                    id,
                    class: materialization.class,
                });
            }
        }

        Ok(Self {
            id,
            role,
            content_identity,
            layout_identity,
            raw_bits,
            serialized_bits,
            serialized_alignment_bits,
            lifetime,
            reconstruction_role,
            resident_materializations,
        })
    }

    /// Segment identifier.
    pub const fn id(&self) -> PhysicalSegmentId {
        self.id
    }

    /// Semantic accounting role.
    pub const fn role(&self) -> PhysicalSegmentRole {
        self.role
    }

    /// Content identity used to validate sharing.
    pub const fn content_identity(&self) -> ContentIdentity {
        self.content_identity
    }

    /// Layout identity used to validate sharing.
    pub const fn layout_identity(&self) -> LayoutIdentity {
        self.layout_identity
    }

    /// Exact encoded extent before container padding.
    pub const fn raw_bits(&self) -> StorageBits {
        self.raw_bits
    }

    /// Exact serialized extent including padding.
    pub const fn serialized_bits(&self) -> StorageBits {
        self.serialized_bits
    }

    /// Required serialized alignment in bits.
    pub const fn serialized_alignment_bits(&self) -> u64 {
        self.serialized_alignment_bits
    }

    /// Ownership lifetime.
    pub const fn lifetime(&self) -> SegmentLifetime {
        self.lifetime
    }

    /// Reconstruction role.
    pub const fn reconstruction_role(&self) -> ReconstructionRole {
        self.reconstruction_role
    }

    /// Declared resident materializations.
    pub fn resident_materializations(&self) -> &[ResidentMaterialization] {
        &self.resident_materializations
    }

    fn resident_bits(
        &self,
        class: MaterializationClass,
    ) -> Result<StorageBits, PhysicalAccountingError> {
        self.resident_materializations
            .iter()
            .find(|materialization| materialization.class == class)
            .map(|materialization| materialization.bits)
            .ok_or(PhysicalAccountingError::UnavailableMaterialization {
                id: self.id,
                class,
            })
    }
}

/// A reference to an owned segment, carrying the identity facts required to
/// prove that the reference denotes exactly the owner's physical object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalSegmentReference {
    id: PhysicalSegmentId,
    content_identity: ContentIdentity,
    layout_identity: LayoutIdentity,
    lifetime: SegmentLifetime,
}

impl PhysicalSegmentReference {
    /// Construct a checked-by-scope reference declaration.
    pub const fn new(
        id: PhysicalSegmentId,
        content_identity: ContentIdentity,
        layout_identity: LayoutIdentity,
        lifetime: SegmentLifetime,
    ) -> Self {
        Self {
            id,
            content_identity,
            layout_identity,
            lifetime,
        }
    }

    /// Referenced segment identifier.
    pub const fn id(&self) -> PhysicalSegmentId {
        self.id
    }
}

/// Owned or referenced segment use inside an accounting scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SegmentUse {
    /// Define the unique physical owner of a segment.
    Own(PhysicalSegment),
    /// Reference an already-owned/shared segment without adding storage.
    Reference(PhysicalSegmentReference),
}

/// Exact physical accounting scope.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PhysicalAccountingScope {
    uses: Vec<SegmentUse>,
}

impl PhysicalAccountingScope {
    /// Construct an empty scope.
    pub const fn new() -> Self {
        Self { uses: Vec::new() }
    }

    /// Add a unique owner declaration.
    pub fn own(&mut self, segment: PhysicalSegment) {
        self.uses.push(SegmentUse::Own(segment));
    }

    /// Add a shared-segment reference.
    pub fn reference(&mut self, reference: PhysicalSegmentReference) {
        self.uses.push(SegmentUse::Reference(reference));
    }

    /// All declarations in insertion order.
    pub fn uses(&self) -> &[SegmentUse] {
        &self.uses
    }

    /// Validate ownership and reference consistency deterministically.
    pub fn validate(&self) -> Result<(), PhysicalAccountingError> {
        self.owners()?;
        Ok(())
    }

    /// Exact serialized size, counting every owned segment once regardless of
    /// how many references point to it.
    pub fn serialized_bits(&self) -> Result<StorageBits, PhysicalAccountingError> {
        let owners = self.owners()?;
        checked_sum(owners.values().map(|segment| segment.serialized_bits.get()))
    }

    /// Exact resident size for one materialization class, counting shared
    /// segments exactly once.
    pub fn resident_bits(
        &self,
        class: MaterializationClass,
    ) -> Result<StorageBits, PhysicalAccountingError> {
        let owners = self.owners()?;
        let mut total = 0u64;
        for segment in owners.values() {
            total = total
                .checked_add(segment.resident_bits(class)?.get())
                .ok_or(PhysicalAccountingError::StorageSizeOverflow)?;
        }
        Ok(StorageBits::new(total))
    }

    /// Exact serialized bits/value rate as an integer rational pair.
    pub fn serialized_rate(
        &self,
        logical_values: u64,
    ) -> Result<EffectiveBitsRate, PhysicalAccountingError> {
        EffectiveBitsRate::new(self.serialized_bits()?, logical_values)
    }

    /// Exact resident bits/value rate as an integer rational pair.
    pub fn resident_rate(
        &self,
        class: MaterializationClass,
        logical_values: u64,
    ) -> Result<EffectiveBitsRate, PhysicalAccountingError> {
        EffectiveBitsRate::new(self.resident_bits(class)?, logical_values)
    }

    fn owners(
        &self,
    ) -> Result<BTreeMap<PhysicalSegmentId, &PhysicalSegment>, PhysicalAccountingError> {
        let mut owners = BTreeMap::new();
        for usage in &self.uses {
            if let SegmentUse::Own(segment) = usage {
                if owners.insert(segment.id, segment).is_some() {
                    return Err(PhysicalAccountingError::DuplicateOwner { id: segment.id });
                }
            }
        }

        for usage in &self.uses {
            let SegmentUse::Reference(reference) = usage else {
                continue;
            };
            let owner = owners
                .get(&reference.id)
                .ok_or(PhysicalAccountingError::MissingOwner { id: reference.id })?;
            if owner.content_identity != reference.content_identity
                || owner.layout_identity != reference.layout_identity
                || owner.lifetime != reference.lifetime
            {
                return Err(PhysicalAccountingError::SharedSegmentMismatch {
                    id: reference.id,
                });
            }
        }

        Ok(owners)
    }
}

/// Exact effective-bit rate represented as an unreduced integer fraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectiveBitsRate {
    bits: StorageBits,
    logical_values: u64,
}

impl EffectiveBitsRate {
    /// Construct an exact rate. A zero denominator is rejected.
    pub fn new(
        bits: StorageBits,
        logical_values: u64,
    ) -> Result<Self, PhysicalAccountingError> {
        if logical_values == 0 {
            return Err(PhysicalAccountingError::ZeroLogicalValueCount);
        }
        Ok(Self {
            bits,
            logical_values,
        })
    }

    /// Exact physical-bit numerator.
    pub const fn bits(&self) -> StorageBits {
        self.bits
    }

    /// Exact logical-value denominator.
    pub const fn logical_values(&self) -> u64 {
        self.logical_values
    }
}

/// Failure while validating or aggregating physical storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PhysicalAccountingError {
    /// Alignment must be a non-zero power of two in bits.
    InvalidAlignment { alignment_bits: u64 },
    /// Serialized physical extent cannot be below its encoded raw extent.
    SerializedBelowRaw { id: PhysicalSegmentId },
    /// A resident materialization cannot be below its encoded raw extent.
    ResidentBelowRaw {
        id: PhysicalSegmentId,
        class: MaterializationClass,
    },
    /// One segment declared the same resident materialization class twice.
    DuplicateMaterialization {
        id: PhysicalSegmentId,
        class: MaterializationClass,
    },
    /// A zero-bit segment lacks explicit reconstruction semantics.
    ZeroBitWithoutReconstruction { id: PhysicalSegmentId },
    /// An implicit-zero segment incorrectly owns physical storage.
    ImplicitZeroHasStorage { id: PhysicalSegmentId },
    /// Two owners attempted to define the same physical segment identifier.
    DuplicateOwner { id: PhysicalSegmentId },
    /// A reference has no unique owner in the accounting scope.
    MissingOwner { id: PhysicalSegmentId },
    /// A reference disagrees with its owner's content/layout/lifetime identity.
    SharedSegmentMismatch { id: PhysicalSegmentId },
    /// A resident size was requested for a materialization not declared by a
    /// segment.
    UnavailableMaterialization {
        id: PhysicalSegmentId,
        class: MaterializationClass,
    },
    /// Checked physical bit aggregation overflowed `u64`.
    StorageSizeOverflow,
    /// Effective bits/value requires a non-zero logical denominator.
    ZeroLogicalValueCount,
}

impl fmt::Display for PhysicalAccountingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAlignment { alignment_bits } => {
                write!(formatter, "alignment {alignment_bits} bits is not a non-zero power of two")
            }
            Self::SerializedBelowRaw { id } => write!(
                formatter,
                "segment {} serialized extent is below raw encoded extent",
                id.get()
            ),
            Self::ResidentBelowRaw { id, class } => write!(
                formatter,
                "segment {} resident materialization {} is below raw encoded extent",
                id.get(),
                class.get()
            ),
            Self::DuplicateMaterialization { id, class } => write!(
                formatter,
                "segment {} declares resident materialization {} more than once",
                id.get(),
                class.get()
            ),
            Self::ZeroBitWithoutReconstruction { id } => write!(
                formatter,
                "segment {} has zero serialized bits without explicit reconstruction semantics",
                id.get()
            ),
            Self::ImplicitZeroHasStorage { id } => write!(
                formatter,
                "implicit-zero segment {} must not own physical storage",
                id.get()
            ),
            Self::DuplicateOwner { id } => {
                write!(formatter, "segment {} has more than one owner", id.get())
            }
            Self::MissingOwner { id } => {
                write!(formatter, "segment {} is referenced without an owner", id.get())
            }
            Self::SharedSegmentMismatch { id } => write!(
                formatter,
                "segment {} reference does not match the owner's content/layout/lifetime",
                id.get()
            ),
            Self::UnavailableMaterialization { id, class } => write!(
                formatter,
                "segment {} has no resident materialization {}",
                id.get(),
                class.get()
            ),
            Self::StorageSizeOverflow => write!(formatter, "physical storage size overflows u64 bits"),
            Self::ZeroLogicalValueCount => {
                write!(formatter, "effective bits/value denominator must be non-zero")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PhysicalAccountingError {}

fn validate_alignment(alignment_bits: u64) -> Result<(), PhysicalAccountingError> {
    if alignment_bits == 0 || !alignment_bits.is_power_of_two() {
        return Err(PhysicalAccountingError::InvalidAlignment { alignment_bits });
    }
    Ok(())
}

fn checked_sum(values: impl Iterator<Item = u64>) -> Result<StorageBits, PhysicalAccountingError> {
    let mut total = 0u64;
    for value in values {
        total = total
            .checked_add(value)
            .ok_or(PhysicalAccountingError::StorageSizeOverflow)?;
    }
    Ok(StorageBits::new(total))
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOST: MaterializationClass = MaterializationClass::new(1);

    fn stored_segment(
        id: u32,
        role: PhysicalSegmentRole,
        content: u64,
        layout: u64,
        raw_bits: u64,
        serialized_bits: u64,
        resident_bits: u64,
    ) -> PhysicalSegment {
        let resident = ResidentMaterialization::new(
            HOST,
            StorageBits::new(resident_bits),
            512,
        )
        .unwrap();
        PhysicalSegment::new(
            PhysicalSegmentId::new(id),
            role,
            ContentIdentity::new(content),
            LayoutIdentity::new(layout),
            StorageBits::new(raw_bits),
            StorageBits::new(serialized_bits),
            64,
            SegmentLifetime::ModelStatic,
            ReconstructionRole::Stored,
            Vec::from([resident]),
        )
        .unwrap()
    }

    #[test]
    fn packed_payload_counts_serialized_and_resident_overhead_exactly() {
        let mut scope = PhysicalAccountingScope::new();
        scope.own(stored_segment(
            0,
            PhysicalSegmentRole::Payload,
            10,
            20,
            512,
            512,
            512,
        ));
        scope.own(stored_segment(
            1,
            PhysicalSegmentRole::Scale,
            11,
            21,
            128,
            128,
            512,
        ));

        assert_eq!(scope.serialized_bits().unwrap().get(), 640);
        assert_eq!(scope.resident_bits(HOST).unwrap().get(), 1024);
        assert_eq!(scope.serialized_rate(256).unwrap().bits().get(), 640);
        assert_eq!(scope.serialized_rate(256).unwrap().logical_values(), 256);
    }

    #[test]
    fn shared_codebook_is_counted_exactly_once() {
        let mut scope = PhysicalAccountingScope::new();
        scope.own(stored_segment(
            0,
            PhysicalSegmentRole::Index,
            100,
            200,
            4096,
            4096,
            4096,
        ));
        scope.own(stored_segment(
            1,
            PhysicalSegmentRole::Index,
            101,
            200,
            4096,
            4096,
            4096,
        ));
        let codebook = stored_segment(
            2,
            PhysicalSegmentRole::Codebook,
            500,
            900,
            4096,
            4096,
            4096,
        );
        scope.own(codebook.clone());
        let shared = PhysicalSegmentReference::new(
            codebook.id(),
            codebook.content_identity(),
            codebook.layout_identity(),
            codebook.lifetime(),
        );
        scope.reference(shared);
        scope.reference(shared);

        assert_eq!(scope.serialized_bits().unwrap().get(), 12_288);
        let rate = scope.serialized_rate(2_048).unwrap();
        assert_eq!(rate.bits().get(), 12_288);
        assert_eq!(rate.logical_values(), 2_048);
    }

    #[test]
    fn missing_owner_and_identity_mismatch_are_rejected() {
        let mut missing = PhysicalAccountingScope::new();
        missing.reference(PhysicalSegmentReference::new(
            PhysicalSegmentId::new(9),
            ContentIdentity::new(1),
            LayoutIdentity::new(2),
            SegmentLifetime::ModelStatic,
        ));
        assert_eq!(
            missing.validate(),
            Err(PhysicalAccountingError::MissingOwner {
                id: PhysicalSegmentId::new(9)
            })
        );

        let owner = stored_segment(
            0,
            PhysicalSegmentRole::Codebook,
            1,
            2,
            64,
            64,
            64,
        );
        let mut mismatch = PhysicalAccountingScope::new();
        mismatch.own(owner.clone());
        mismatch.reference(PhysicalSegmentReference::new(
            owner.id(),
            ContentIdentity::new(999),
            owner.layout_identity(),
            owner.lifetime(),
        ));
        assert_eq!(
            mismatch.validate(),
            Err(PhysicalAccountingError::SharedSegmentMismatch { id: owner.id() })
        );
    }

    #[test]
    fn duplicate_owner_is_rejected_even_when_definition_matches() {
        let segment = stored_segment(
            0,
            PhysicalSegmentRole::Payload,
            1,
            2,
            64,
            64,
            64,
        );
        let mut scope = PhysicalAccountingScope::new();
        scope.own(segment.clone());
        scope.own(segment);
        assert_eq!(
            scope.validate(),
            Err(PhysicalAccountingError::DuplicateOwner {
                id: PhysicalSegmentId::new(0)
            })
        );
    }

    #[test]
    fn zero_bit_storage_requires_explicit_reconstruction() {
        let rejected = PhysicalSegment::new(
            PhysicalSegmentId::new(0),
            PhysicalSegmentRole::Payload,
            ContentIdentity::new(0),
            LayoutIdentity::new(0),
            StorageBits::new(0),
            StorageBits::new(0),
            8,
            SegmentLifetime::Block,
            ReconstructionRole::Stored,
            Vec::new(),
        );
        assert_eq!(
            rejected,
            Err(PhysicalAccountingError::ZeroBitWithoutReconstruction {
                id: PhysicalSegmentId::new(0)
            })
        );

        let implicit = PhysicalSegment::new(
            PhysicalSegmentId::new(1),
            PhysicalSegmentRole::Payload,
            ContentIdentity::new(0),
            LayoutIdentity::new(0),
            StorageBits::new(0),
            StorageBits::new(0),
            8,
            SegmentLifetime::Block,
            ReconstructionRole::ImplicitZero,
            Vec::new(),
        )
        .unwrap();
        let mut scope = PhysicalAccountingScope::new();
        scope.own(implicit);
        assert_eq!(scope.serialized_bits().unwrap().get(), 0);
    }

    #[test]
    fn checked_aggregation_rejects_overflow() {
        let mut scope = PhysicalAccountingScope::new();
        scope.own(stored_segment(
            0,
            PhysicalSegmentRole::Payload,
            1,
            1,
            u64::MAX,
            u64::MAX,
            u64::MAX,
        ));
        scope.own(stored_segment(
            1,
            PhysicalSegmentRole::Metadata,
            2,
            2,
            1,
            1,
            1,
        ));
        assert_eq!(
            scope.serialized_bits(),
            Err(PhysicalAccountingError::StorageSizeOverflow)
        );
    }

    #[test]
    fn effective_rate_rejects_zero_denominator() {
        assert_eq!(
            EffectiveBitsRate::new(StorageBits::new(1), 0),
            Err(PhysicalAccountingError::ZeroLogicalValueCount)
        );
    }
}
