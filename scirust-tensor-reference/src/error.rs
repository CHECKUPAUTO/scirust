//! Typed failures for Reference-kernel generation and decoding.

use std::fmt;

use scirust_compute::{ComputeError, DType, KernelFormat};
use scirust_tensor_compile::LogicalKernelId;

use crate::format::ReferenceOpcode;

/// A failure while generating Reference-format artefacts from a `LoweredPlan`.
///
/// No variant carries a bare [`String`] as its only payload: every failure
/// names the kernel and the specific invariant it violates.
///
/// Several variants guard invariants that a [`scirust_tensor_compile::LoweredPlan`]
/// produced by `KernelLowerer::lower` already establishes today (kernels are
/// positioned by construction, dtypes are already uniform, `Scalar` only
/// constructs `F32` values). They exist so a future relaxation of those
/// upstream guarantees fails loudly here instead of silently producing a wrong
/// artefact, and they are honestly not reachable — or covered by tests — through
/// the current public API. Each such variant says so in its own documentation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReferenceGenerationError {
    /// The plan carries more kernels than a `u32` logical identifier can index.
    ///
    /// Unreachable in practice: producing that many kernels would already have
    /// failed inside `scirust_tensor_compile` while assigning `LogicalKernelId`s.
    TooManyKernels { count: usize },
    /// Two kernels in the plan claim the same [`LogicalKernelId`].
    ///
    /// Unreachable through a `LoweredPlan` produced by `KernelLowerer::lower`,
    /// which assigns ids sequentially by construction. Retained as a defensive
    /// check against a future relaxation of that guarantee.
    DuplicateLogicalKernel { kernel: LogicalKernelId },
    /// A [`LogicalKernelId`] expected at some position is absent from the plan.
    ///
    /// Unreachable by a pigeonhole argument: once every kernel in an `n`-kernel
    /// plan is confirmed both in-range and pairwise distinct, the `n` ids must
    /// already cover `0..n` exactly, so none can be missing.
    MissingLogicalKernel { expected: LogicalKernelId },
    /// A kernel's [`LogicalKernelId`] does not match its position in the plan.
    ///
    /// Unreachable through a `LoweredPlan` produced by `KernelLowerer::lower`
    /// for the same reason as [`Self::DuplicateLogicalKernel`].
    OutOfOrderLogicalKernel {
        expected: LogicalKernelId,
        found: LogicalKernelId,
    },
    /// The kernel's family has no Reference opcode in this format version.
    UnsupportedKernelFamily { kernel: LogicalKernelId },
    /// The kernel's element type is not `DType::F32`, the only type Reference
    /// v1.0 supports.
    UnsupportedDType {
        kernel: LogicalKernelId,
        dtype: DType,
    },
    /// The kernel's operands and result do not all share one element type.
    ///
    /// Guards an invariant `scirust_tensor_compile`'s lowering already
    /// establishes; re-checked here rather than trusted.
    MixedDTypes { kernel: LogicalKernelId },
    /// A `Scale` kernel's factor does not carry the kernel's element type.
    ///
    /// Unreachable today: `Scalar` has only an `f32` constructor, so its
    /// `dtype()` is always `DType::F32`, matching the only type this format
    /// version supports. Re-checked rather than assumed.
    ScaleFactorTypeMismatch { kernel: LogicalKernelId },
    /// A `Scale` kernel's factor bit pattern does not fit in `u32`.
    ///
    /// Unreachable today: a `DType::F32` `Scalar` always stores its bits in the
    /// lower 32 of its 64-bit representation.
    ScaleFactorBitsOverflow { kernel: LogicalKernelId },
    /// The kernel's operand count does not match its opcode's fixed arity.
    ///
    /// Guards an invariant `scirust_tensor_compile`'s lowering already
    /// establishes; re-checked here rather than trusted.
    UnexpectedOperandCount {
        kernel: LogicalKernelId,
        expected: usize,
        actual: usize,
    },
    /// An operand or result rank exceeds [`crate::REFERENCE_MAX_RANK`].
    RankTooLarge {
        kernel: LogicalKernelId,
        rank: usize,
        limit: u16,
    },
    /// A dimension does not fit in the wire format's `u64`.
    ///
    /// Unreachable on any platform Rust actually targets: `usize` never exceeds
    /// 64 bits there. Retained for portability to a hypothetical wider target.
    DimensionOverflow { kernel: LogicalKernelId },
    /// The element count does not fit in the wire format's `u64`, or overflows
    /// while being computed.
    ///
    /// Unreachable on any platform Rust actually targets, for the same reason
    /// as [`Self::DimensionOverflow`].
    ElementCountOverflow { kernel: LogicalKernelId },
    /// A permutation axis does not fit in the wire format's `u16`.
    ///
    /// Unreachable given [`crate::REFERENCE_MAX_RANK`] `<= 32`: every valid axis
    /// index is already far smaller than `u16::MAX`.
    AxisIndexOverflow { kernel: LogicalKernelId },
    /// Building the final [`scirust_compute::KernelModule`] failed.
    InvalidKernelModule {
        kernel: LogicalKernelId,
        source: ComputeError,
    },
}

impl fmt::Display for ReferenceGenerationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::TooManyKernels { count } =>
            {
                write!(
                    formatter,
                    "plan has {count} kernels, too many for a u32 logical id"
                )
            },
            Self::DuplicateLogicalKernel { kernel } =>
            {
                write!(
                    formatter,
                    "logical kernel {} appears more than once in the plan",
                    kernel.get()
                )
            },
            Self::MissingLogicalKernel { expected } =>
            {
                write!(
                    formatter,
                    "logical kernel {} is absent from the plan",
                    expected.get()
                )
            },
            Self::OutOfOrderLogicalKernel { expected, found } =>
            {
                write!(
                    formatter,
                    "expected logical kernel {} at this position, found {}",
                    expected.get(),
                    found.get()
                )
            },
            Self::UnsupportedKernelFamily { kernel } =>
            {
                write!(
                    formatter,
                    "logical kernel {} has no Reference opcode in this format version",
                    kernel.get()
                )
            },
            Self::UnsupportedDType { kernel, dtype } =>
            {
                write!(
                    formatter,
                    "logical kernel {} uses {dtype:?}, but Reference v1.0 supports only F32",
                    kernel.get()
                )
            },
            Self::MixedDTypes { kernel } =>
            {
                write!(
                    formatter,
                    "logical kernel {}'s operands and result do not share one element type",
                    kernel.get()
                )
            },
            Self::ScaleFactorTypeMismatch { kernel } =>
            {
                write!(
                    formatter,
                    "logical kernel {}'s scale factor does not carry the kernel's element type",
                    kernel.get()
                )
            },
            Self::ScaleFactorBitsOverflow { kernel } =>
            {
                write!(
                    formatter,
                    "logical kernel {}'s scale factor bit pattern does not fit in u32",
                    kernel.get()
                )
            },
            Self::UnexpectedOperandCount {
                kernel,
                expected,
                actual,
            } =>
            {
                write!(
                    formatter,
                    "logical kernel {} expects {expected} operands but carries {actual}",
                    kernel.get()
                )
            },
            Self::RankTooLarge {
                kernel,
                rank,
                limit,
            } =>
            {
                write!(
                    formatter,
                    "logical kernel {} has rank {rank}, exceeding the Reference limit of {limit}",
                    kernel.get()
                )
            },
            Self::DimensionOverflow { kernel } =>
            {
                write!(
                    formatter,
                    "logical kernel {} has a dimension that does not fit in u64",
                    kernel.get()
                )
            },
            Self::ElementCountOverflow { kernel } =>
            {
                write!(
                    formatter,
                    "logical kernel {}'s element count does not fit in u64",
                    kernel.get()
                )
            },
            Self::AxisIndexOverflow { kernel } =>
            {
                write!(
                    formatter,
                    "logical kernel {} has a permutation axis that does not fit in u16",
                    kernel.get()
                )
            },
            Self::InvalidKernelModule { kernel, source } =>
            {
                write!(
                    formatter,
                    "logical kernel {} produced an invalid kernel module: {source}",
                    kernel.get()
                )
            },
        }
    }
}

impl std::error::Error for ReferenceGenerationError {}

/// A failure while decoding a byte stream as a Reference-format artefact.
///
/// The decoder only parses, validates and reconstructs a typed artefact; it
/// never executes anything. No variant carries a bare [`String`] as its only
/// payload.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReferenceDecodeError {
    /// The stream does not start with [`crate::REFERENCE_MAGIC`].
    InvalidMagic,
    /// The stream's major version differs from the one this decoder implements.
    IncompatibleMajorVersion { found: u16, supported: u16 },
    /// The stream's minor version differs from the one this decoder implements.
    ///
    /// This decoder accepts exactly one minor version; it does not yet
    /// implement a "accept any minor `<=` supported" policy. See the format
    /// documentation for the versioning policy this will follow once a second
    /// minor version exists.
    UnsupportedMinorVersion { found: u16, supported: u16 },
    /// The opcode field does not name a known [`ReferenceOpcode`].
    UnknownOpcode { code: u16 },
    /// The dtype tag does not name any `DType` this format's tag table knows.
    UnknownDType { tag: u16 },
    /// The dtype tag names a `DType` SciRust knows, but Reference v1.0 does not
    /// support it.
    UnsupportedDType { dtype: DType },
    /// The attribute tag does not name a known attribute kind.
    UnknownAttributeTag { tag: u16 },
    /// The attribute tag is inconsistent with the opcode that precedes it.
    AttributeOpcodeMismatch { opcode: ReferenceOpcode },
    /// The stream ended before a required field could be read.
    TruncatedInput { needed: usize, available: usize },
    /// Bytes remain after a complete, valid artefact was parsed.
    TrailingBytes { remaining: usize },
    /// An operand or result rank exceeds [`crate::REFERENCE_MAX_RANK`].
    RankTooLarge { rank: u16, limit: u16 },
    /// The result count is not exactly `1`.
    InvalidResultCount { found: u16 },
    /// The operand count does not match the opcode's fixed arity.
    UnexpectedOperandCount {
        opcode: ReferenceOpcode,
        expected: usize,
        actual: usize,
    },
    /// A layout's declared element count overflows while being recomputed from
    /// its dimensions.
    ElementCountOverflow,
    /// A layout's declared element count disagrees with the product of its
    /// dimensions, or (for `ShapeCopy`) an operand's element count disagrees
    /// with the result's.
    ElementCountMismatch,
    /// A `Permute` axis list is not a valid permutation of its rank: wrong
    /// length, an axis repeated, or an axis out of bounds.
    InvalidPermutation,
    /// An operand or result shape is inconsistent with the opcode's semantics:
    /// an element-wise operand does not match the result shape, or a `Permute`
    /// result does not satisfy `output.shape[i] == input.shape[permutation[i]]`.
    ShapeMismatch,
}

impl fmt::Display for ReferenceDecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidMagic =>
            {
                formatter.write_str("stream does not start with the Reference magic")
            },
            Self::IncompatibleMajorVersion { found, supported } =>
            {
                write!(
                    formatter,
                    "stream major version {found} is incompatible with supported version {supported}"
                )
            },
            Self::UnsupportedMinorVersion { found, supported } =>
            {
                write!(
                    formatter,
                    "stream minor version {found} is not supported; this decoder implements version {supported}"
                )
            },
            Self::UnknownOpcode { code } => write!(formatter, "opcode 0x{code:04X} is unknown"),
            Self::UnknownDType { tag } => write!(formatter, "dtype tag 0x{tag:04X} is unknown"),
            Self::UnsupportedDType { dtype } =>
            {
                write!(
                    formatter,
                    "{dtype:?} is known to SciRust but not supported by Reference v1.0"
                )
            },
            Self::UnknownAttributeTag { tag } =>
            {
                write!(formatter, "attribute tag 0x{tag:04X} is unknown")
            },
            Self::AttributeOpcodeMismatch { opcode } =>
            {
                write!(
                    formatter,
                    "attribute tag is inconsistent with opcode {opcode:?}"
                )
            },
            Self::TruncatedInput { needed, available } =>
            {
                write!(
                    formatter,
                    "stream ended: needed {needed} more bytes, {available} available"
                )
            },
            Self::TrailingBytes { remaining } =>
            {
                write!(
                    formatter,
                    "{remaining} unexpected trailing byte(s) after a complete artefact"
                )
            },
            Self::RankTooLarge { rank, limit } =>
            {
                write!(
                    formatter,
                    "rank {rank} exceeds the Reference limit of {limit}"
                )
            },
            Self::InvalidResultCount { found } =>
            {
                write!(
                    formatter,
                    "result count {found} is invalid; Reference v1.0 requires exactly 1"
                )
            },
            Self::UnexpectedOperandCount {
                opcode,
                expected,
                actual,
            } =>
            {
                write!(
                    formatter,
                    "opcode {opcode:?} expects {expected} operands but the stream carries {actual}"
                )
            },
            Self::ElementCountOverflow =>
            {
                formatter.write_str("a layout's element count overflows u64")
            },
            Self::ElementCountMismatch =>
            {
                formatter.write_str("a layout's declared element count is inconsistent")
            },
            Self::InvalidPermutation =>
            {
                formatter.write_str("permutation is not a valid axis rearrangement")
            },
            Self::ShapeMismatch =>
            {
                formatter.write_str("operand or result shape is inconsistent with the opcode")
            },
        }
    }
}

impl std::error::Error for ReferenceDecodeError {}

/// A failure while preparing or executing a Reference kernel on the CPU.
///
/// No variant carries a bare [`String`] as a free-form message: `String` appears
/// only in [`Self::EntryPointMismatch`], where both sides of a concrete
/// comparison are reported.
///
/// Four variants are **defensive** and not reachable through the current public
/// API — each says so in its own documentation. They exist so a future
/// relaxation of an upstream guarantee fails loudly with a typed error instead
/// of panicking or silently computing the wrong result. This crate contains no
/// `panic!`, `unwrap()`, `expect()` or unchecked indexing on any preparation or
/// execution path, so these variants are the mechanism by which a "cannot
/// happen" state is reported rather than hit.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReferenceExecutionError {
    /// The module is not [`scirust_compute::KernelFormat::Reference`]: WGSL,
    /// PTX, SPIR-V and any future format are rejected here.
    WrongKernelFormat { found: KernelFormat },
    /// The module's `code` is not a valid Reference-format stream.
    InvalidEncoding(ReferenceDecodeError),
    /// The module's `entry_point` differs from the name canonically derived from
    /// the decoded artefact's `LogicalKernelId`.
    EntryPointMismatch { expected: String, found: String },
    /// The opcode has no CPU implementation in this interpreter.
    ///
    /// Reserved for a future opcode. Not constructed today: this crate matches
    /// [`ReferenceOpcode`] exhaustively, so adding a variant to that enum
    /// produces a **compile error** here rather than a runtime rejection —
    /// a stronger guarantee than this variant provides, and the reason it stays
    /// unreachable.
    UnsupportedOpcode { opcode: ReferenceOpcode },
    /// The opcode needs a transcendental function this crate cannot evaluate
    /// with a cross-platform bit-identical result.
    ///
    /// Returned for `Exp` and `Log`. See the crate documentation for why they
    /// are rejected rather than approximated.
    DeterministicMathUnavailable { opcode: ReferenceOpcode },
    /// The kernel's element type is not `DType::F32`.
    UnsupportedDType { dtype: DType },
    /// The invocation supplied the wrong number of operand slices.
    OperandCountMismatch { expected: usize, actual: usize },
    /// An operand slice's length does not match the length the kernel declares.
    OperandLengthMismatch {
        operand_index: usize,
        expected: usize,
        actual: usize,
    },
    /// The output slice's length does not match the length the kernel declares.
    OutputLengthMismatch { expected: usize, actual: usize },
    /// The artefact's attribute payload is inconsistent with its opcode.
    ///
    /// Defensive: `codec` and `generator` already establish this coherence
    /// before a [`crate::ReferenceKernelArtifact`] can exist.
    AttributeMismatch { opcode: ReferenceOpcode },
    /// A `u64` dimension or element count does not fit in the host's `usize`.
    ///
    /// Defensive: unreachable on any platform Rust targets today, where `usize`
    /// is at least 32 and at most 64 bits and the value was already produced
    /// from a `usize` upstream.
    DimensionConversionOverflow { dimension: u64 },
    /// A stride or linear index computation overflowed `usize`.
    ///
    /// Defensive: the element count that bounds every index was itself already
    /// converted from a `usize` upstream, so no in-range index can overflow.
    IndexOverflow,
    /// An internal bounds check failed while indexing an already-validated
    /// buffer, shape or stride table.
    ///
    /// Defensive: reported instead of panicking. Every access on an execution
    /// path goes through `get`/`get_mut`, so a broken internal invariant
    /// surfaces as this error rather than as an abort.
    InternalIndexOutOfBounds,
}

impl fmt::Display for ReferenceExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::WrongKernelFormat { found } =>
            {
                write!(
                    formatter,
                    "kernel module has format {found:?}; the Reference interpreter requires Reference"
                )
            },
            Self::InvalidEncoding(error) =>
            {
                write!(
                    formatter,
                    "kernel module code is not a valid Reference stream: {error}"
                )
            },
            Self::EntryPointMismatch { expected, found } =>
            {
                write!(
                    formatter,
                    "kernel module entry point is {found:?} but the artefact derives {expected:?}"
                )
            },
            Self::UnsupportedOpcode { opcode } =>
            {
                write!(formatter, "opcode {opcode:?} has no CPU implementation")
            },
            Self::DeterministicMathUnavailable { opcode } =>
            {
                write!(
                    formatter,
                    "opcode {opcode:?} needs a transcendental function with no cross-platform bit-identical implementation available to this crate"
                )
            },
            Self::UnsupportedDType { dtype } =>
            {
                write!(
                    formatter,
                    "the Reference interpreter executes F32 only, not {dtype:?}"
                )
            },
            Self::OperandCountMismatch { expected, actual } =>
            {
                write!(
                    formatter,
                    "kernel expects {expected} operand(s) but the invocation supplied {actual}"
                )
            },
            Self::OperandLengthMismatch {
                operand_index,
                expected,
                actual,
            } =>
            {
                write!(
                    formatter,
                    "operand {operand_index} must hold {expected} element(s) but holds {actual}"
                )
            },
            Self::OutputLengthMismatch { expected, actual } =>
            {
                write!(
                    formatter,
                    "output must hold {expected} element(s) but holds {actual}"
                )
            },
            Self::AttributeMismatch { opcode } =>
            {
                write!(
                    formatter,
                    "attribute payload is inconsistent with opcode {opcode:?}"
                )
            },
            Self::DimensionConversionOverflow { dimension } =>
            {
                write!(formatter, "dimension {dimension} does not fit in usize")
            },
            Self::IndexOverflow =>
            {
                formatter.write_str("a stride or linear index computation overflowed usize")
            },
            Self::InternalIndexOutOfBounds =>
            {
                formatter.write_str("an internal bounds check failed on a validated buffer")
            },
        }
    }
}

impl core::error::Error for ReferenceExecutionError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self
        {
            Self::InvalidEncoding(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ReferenceDecodeError> for ReferenceExecutionError {
    fn from(error: ReferenceDecodeError) -> Self {
        Self::InvalidEncoding(error)
    }
}
