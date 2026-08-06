//! Wire-format constants: magic, version, rank limit, opcodes and the stable
//! `DType` tag table.

use scirust_compute::DType;

/// Domain-separation prefix at the start of every Reference-format stream.
pub const REFERENCE_MAGIC: &[u8; 25] = b"SCIRUST-REFERENCE-KERNEL\0";

/// Largest rank a Reference-format layout may declare.
///
/// Bounds the decoder's rank-dependent allocation against untrusted input; it
/// is not a mathematical limit on tensor rank.
pub const REFERENCE_MAX_RANK: u16 = 32;

/// Attribute-tag constants (`u16`), stable once published.
pub(crate) const ATTRIBUTE_TAG_NONE: u16 = 0x0000;
pub(crate) const ATTRIBUTE_TAG_SCALE: u16 = 0x0001;
pub(crate) const ATTRIBUTE_TAG_PERMUTE: u16 = 0x0002;

/// Version of the Reference wire format.
///
/// # Versioning policy
///
/// * An incompatible structural change (reordering, removing or resizing a
///   field, redefining an opcode or a `DType` tag) is a new **major** version.
/// * An explicitly supported extension (a new opcode, a new attribute, a wider
///   supported `DType` set, a raised [`REFERENCE_MAX_RANK`]) is a new **minor**
///   version.
///
/// Only version 1.0 exists today, so this policy is a stated intent, not a
/// demonstrated guarantee: the decoder in this crate accepts exactly
/// `major == 1, minor == 0` and rejects every other value, including a lower
/// minor version. A future 1.1 decoder — once that format genuinely exists —
/// is expected to accept both 1.0 and 1.1 streams explicitly; this crate does
/// not pre-implement that "accept any minor `<=` supported" behaviour ahead of
/// there being a second version to accept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceFormatVersion {
    major: u16,
    minor: u16,
}

impl ReferenceFormatVersion {
    pub const fn new(major: u16, minor: u16) -> Self {
        Self { major, minor }
    }

    pub const fn major(self) -> u16 {
        self.major
    }

    pub const fn minor(self) -> u16 {
        self.minor
    }
}

/// The single Reference format version this crate generates and decodes.
pub const REFERENCE_FORMAT_VERSION: ReferenceFormatVersion = ReferenceFormatVersion::new(1, 0);

/// Closed set of Reference-format operations for this format version.
///
/// Exactly mirrors the kernel families `scirust_tensor_compile::KernelLowerer`
/// produces today. Opcode values are published and never reassigned; adding a
/// new opcode is a minor version bump, never a silent reuse of a retired code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ReferenceOpcode {
    Relu,
    Exp,
    Log,
    Scale,
    Add,
    Sub,
    Mul,
    Div,
    ShapeCopy,
    Permute,
}

impl ReferenceOpcode {
    /// Stable wire value. Never reassigned to a different operation.
    pub const fn code(self) -> u16 {
        match self
        {
            Self::Relu => 0x0001,
            Self::Exp => 0x0002,
            Self::Log => 0x0003,
            Self::Scale => 0x0004,
            Self::Add => 0x0010,
            Self::Sub => 0x0011,
            Self::Mul => 0x0012,
            Self::Div => 0x0013,
            Self::ShapeCopy => 0x0020,
            Self::Permute => 0x0030,
        }
    }

    /// Recovers the opcode named by a wire value, or `None` if unknown.
    pub const fn from_code(code: u16) -> Option<Self> {
        match code
        {
            0x0001 => Some(Self::Relu),
            0x0002 => Some(Self::Exp),
            0x0003 => Some(Self::Log),
            0x0004 => Some(Self::Scale),
            0x0010 => Some(Self::Add),
            0x0011 => Some(Self::Sub),
            0x0012 => Some(Self::Mul),
            0x0013 => Some(Self::Div),
            0x0020 => Some(Self::ShapeCopy),
            0x0030 => Some(Self::Permute),
            _ => None,
        }
    }

    /// Fixed operand arity of this opcode, mirroring
    /// `scirust_tensor_ir::Operation::expected_arity` for the operations this
    /// format version covers.
    pub const fn operand_count(self) -> u16 {
        match self
        {
            Self::Add | Self::Sub | Self::Mul | Self::Div => 2,
            Self::Relu | Self::Exp | Self::Log | Self::Scale | Self::ShapeCopy | Self::Permute => 1,
        }
    }
}

/// Maps a `DType` to its stable wire tag.
///
/// Every variant `DType` currently declares has a tag reserved here, even
/// though Reference v1.0 supports only `F32` for generation and decoding.
/// Reserving the full table now means widening the supported set later is a
/// minor version bump touching one list, not a renumbering.
///
/// `DType` is `#[non_exhaustive]`; a variant added to it in the future falls
/// through to `0x0000`, a tag reserved as "unknown to this table" and never
/// assigned to a real `DType`.
pub(crate) const fn dtype_to_tag(dtype: DType) -> u16 {
    match dtype
    {
        DType::Bool => 0x0001,
        DType::U8 => 0x0002,
        DType::I8 => 0x0003,
        DType::U16 => 0x0004,
        DType::I16 => 0x0005,
        DType::F16 => 0x0006,
        DType::Bf16 => 0x0007,
        DType::U32 => 0x0008,
        DType::I32 => 0x0009,
        DType::F32 => 0x000A,
        DType::U64 => 0x000B,
        DType::I64 => 0x000C,
        DType::F64 => 0x000D,
        _ => 0x0000,
    }
}

/// Recovers the `DType` named by a wire tag.
///
/// Returns `None` for a tag this table has never assigned to any `DType` —
/// including `0x0000`, which is reserved and never assigned. A `Some(dtype)`
/// result does not imply Reference v1.0 supports `dtype`; callers must check
/// that separately (`dtype == DType::F32`), so an unknown tag and a known but
/// unsupported `DType` are always distinguishable.
pub(crate) const fn dtype_from_tag(tag: u16) -> Option<DType> {
    match tag
    {
        0x0001 => Some(DType::Bool),
        0x0002 => Some(DType::U8),
        0x0003 => Some(DType::I8),
        0x0004 => Some(DType::U16),
        0x0005 => Some(DType::I16),
        0x0006 => Some(DType::F16),
        0x0007 => Some(DType::Bf16),
        0x0008 => Some(DType::U32),
        0x0009 => Some(DType::I32),
        0x000A => Some(DType::F32),
        0x000B => Some(DType::U64),
        0x000C => Some(DType::I64),
        0x000D => Some(DType::F64),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opcode_round_trips_through_its_code() {
        for opcode in [
            ReferenceOpcode::Relu,
            ReferenceOpcode::Exp,
            ReferenceOpcode::Log,
            ReferenceOpcode::Scale,
            ReferenceOpcode::Add,
            ReferenceOpcode::Sub,
            ReferenceOpcode::Mul,
            ReferenceOpcode::Div,
            ReferenceOpcode::ShapeCopy,
            ReferenceOpcode::Permute,
        ]
        {
            assert_eq!(ReferenceOpcode::from_code(opcode.code()), Some(opcode));
        }
    }

    #[test]
    fn opcode_values_are_stable_and_distinct() {
        let codes = [
            ReferenceOpcode::Relu.code(),
            ReferenceOpcode::Exp.code(),
            ReferenceOpcode::Log.code(),
            ReferenceOpcode::Scale.code(),
            ReferenceOpcode::Add.code(),
            ReferenceOpcode::Sub.code(),
            ReferenceOpcode::Mul.code(),
            ReferenceOpcode::Div.code(),
            ReferenceOpcode::ShapeCopy.code(),
            ReferenceOpcode::Permute.code(),
        ];

        assert_eq!(
            codes,
            [
                0x0001, 0x0002, 0x0003, 0x0004, 0x0010, 0x0011, 0x0012, 0x0013, 0x0020, 0x0030,
            ]
        );
    }

    #[test]
    fn unknown_opcode_is_rejected() {
        assert_eq!(ReferenceOpcode::from_code(0xFFFF), None);
    }

    #[test]
    fn every_known_dtype_round_trips_through_its_tag() {
        for dtype in [
            DType::Bool,
            DType::U8,
            DType::I8,
            DType::U16,
            DType::I16,
            DType::F16,
            DType::Bf16,
            DType::U32,
            DType::I32,
            DType::F32,
            DType::U64,
            DType::I64,
            DType::F64,
        ]
        {
            assert_eq!(dtype_from_tag(dtype_to_tag(dtype)), Some(dtype));
        }
    }

    #[test]
    fn tag_zero_is_reserved_and_unknown() {
        assert_eq!(dtype_from_tag(0x0000), None);
    }

    #[test]
    fn a_completely_unknown_tag_is_distinguishable_from_a_known_unsupported_dtype() {
        // 0xBEEF names no DType in the table at all.
        assert_eq!(dtype_from_tag(0xBEEF), None);
        // 0x0001 (Bool) is known to the table, just not supported by Reference
        // v1.0 — callers must check that separately from `from_tag` succeeding.
        assert_eq!(dtype_from_tag(0x0001), Some(DType::Bool));
    }
}
