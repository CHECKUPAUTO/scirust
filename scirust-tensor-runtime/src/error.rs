//! Typed failures of the Reference plan runtime.
//!
//! Preparation and execution fail through two separate enums, so a caller can
//! tell "this plan can never run here" from "this run did not work out". Both
//! keep a backend [`ComputeError`] as a `source()` rather than flattening it
//! into a message, and neither has a variant that is only a free-form string.

use core::fmt;

use scirust_compute::ComputeError;
use scirust_tensor_compile::{BufferSlot, LogicalBindingId, LogicalKernelId};
use scirust_tensor_ir::NodeId;
use scirust_tensor_reference::{ReferenceGenerationError, ReferenceOpcode};

/// A failure while turning a `LoweredPlan` into a runnable plan.
///
/// Several variants guard invariants that `KernelLowerer` already establishes,
/// and that no public constructor lets a caller violate — `LoweredPlan`'s fields
/// are private and only the lowerer builds one. They are checked anyway rather
/// than assumed, and each such variant says so in its own documentation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlanPreparationError {
    /// A kernel's `LogicalKernelId` is not its position in the plan.
    ///
    /// Defensive: the lowerer interns kernels by position, and
    /// `ReferenceKernelGenerator` re-verifies it.
    NonDenseKernelId {
        position: usize,
        found: LogicalKernelId,
    },
    /// A dispatch names a kernel the plan does not contain.
    UnknownKernelId {
        dispatch_index: usize,
        kernel: LogicalKernelId,
    },
    /// An argument names an external binding the plan does not contain.
    UnknownBindingId {
        dispatch_index: usize,
        binding: LogicalBindingId,
    },
    /// An argument's binding kind disagrees with the plan's binding table.
    BindingKindMismatch {
        dispatch_index: usize,
        binding: LogicalBindingId,
    },
    /// A buffer slot index in `0..=max` is never referenced, so slots are not
    /// dense and cannot index a physical buffer table.
    ///
    /// Defensive: the memory planner allocates slots consecutively from zero.
    NonDenseBufferSlot { slot: BufferSlot },
    /// The dispatch's argument count does not match its kernel's arity plus one
    /// result.
    ArgumentCountMismatch {
        dispatch_index: usize,
        expected: usize,
        actual: usize,
    },
    /// An argument's declared index is not its position.
    ArgumentIndexMismatch {
        dispatch_index: usize,
        position: usize,
        found: u32,
    },
    /// The dispatch has no write argument.
    MissingWriteArgument { dispatch_index: usize },
    /// The dispatch has more than one write argument.
    MultipleWriteArguments { dispatch_index: usize },
    /// The dispatch's write argument is not its last argument.
    WriteArgumentNotLast {
        dispatch_index: usize,
        position: usize,
    },
    /// The dispatch writes to an external input or constant.
    ///
    /// External values are immutable for this runtime: a dispatch result always
    /// lands in an internal buffer slot.
    WriteToExternalValue {
        dispatch_index: usize,
        binding: LogicalBindingId,
    },
    /// An argument's tensor type disagrees with its kernel's signature.
    ArgumentTypeMismatch {
        dispatch_index: usize,
        position: usize,
    },
    /// Two occurrences of one buffer slot declare different tensor types.
    ///
    /// Defensive: the memory planner only reuses a slot for a value whose
    /// tensor type is exactly equal, so all occurrences of a slot agree.
    InconsistentSlotType { slot: BufferSlot },
    /// Two occurrences of one external value declare different tensor types.
    InconsistentExternalType { binding: LogicalBindingId },
    /// A value's tensor type appears nowhere in the plan.
    ///
    /// `LoweredPlan` carries a tensor type only on kernel arguments. An external
    /// value that is *only* a plan output — never consumed by any dispatch —
    /// therefore has no type anywhere, and this runtime rejects the plan instead
    /// of inventing a shape from whatever the caller happens to supply.
    UndeterminedValueType { node: NodeId },
    /// The plan uses an element type this runtime cannot execute.
    UnsupportedDType { node: NodeId },
    /// The backend does not advertise support for `F32`.
    BackendDTypeUnsupported,
    /// The backend advertises a workgroup limit smaller than the canonical
    /// single-element launch this runtime issues.
    BackendWorkgroupTooSmall { max_workgroup_size: [u32; 3] },
    /// A dispatch's declared element count disagrees with its result type.
    DispatchExtentMismatch {
        dispatch_index: usize,
        expected: usize,
        actual: usize,
    },
    /// A dispatch reads a buffer slot no earlier dispatch has written.
    ReadBeforeDefinition {
        dispatch_index: usize,
        slot: BufferSlot,
    },
    /// An argument declares an access kind this runtime does not handle.
    ///
    /// `KernelArgumentAccess` is `#[non_exhaustive]`. Access is converted once,
    /// at preparation, into an internal two-valued form, so a future variant is
    /// rejected here instead of being silently treated as a read or a write on
    /// the execution path.
    UnsupportedArgumentAccess {
        dispatch_index: usize,
        position: usize,
    },
    /// An argument names a source kind this runtime does not handle.
    ///
    /// `KernelArgumentSource` is `#[non_exhaustive]`: a variant added to it in
    /// the future is rejected here rather than silently treated as one of the
    /// kinds this runtime already knows.
    UnsupportedArgumentSource {
        dispatch_index: usize,
        position: usize,
    },
    /// A plan output names a source the plan does not contain, or a source kind
    /// this runtime does not handle.
    UnknownOutputSource { output_index: usize, node: NodeId },
    /// A buffer's byte size overflows `usize`.
    SizeOverflow { elements: usize },
    /// A buffer's byte size exceeds the backend's advertised maximum.
    BufferExceedsBackendLimit { bytes: usize, limit: usize },
    /// The plan uses `Exp` or `Log`, which no bit-reproducible CPU
    /// implementation is available for.
    ///
    /// Rejected here, by the runtime itself, rather than left to whichever
    /// backend happens to be supplied: the set of executable opcodes is a
    /// property of this runtime phase, not of the backend.
    DeterministicMathUnavailable {
        kernel: LogicalKernelId,
        opcode: ReferenceOpcode,
    },
    /// Reference artefact generation failed.
    KernelGeneration { source: ReferenceGenerationError },
    /// The generated artefact set does not match the plan's kernels.
    ///
    /// Defensive: `ReferenceKernelGenerator` preserves the plan's kernel order
    /// and identifiers.
    KernelArtifactMismatch { expected: usize, actual: usize },
    /// Converting an artefact into a `KernelModule` failed.
    KernelModule {
        kernel: LogicalKernelId,
        source: ReferenceGenerationError,
    },
    /// The backend rejected a kernel module.
    KernelCompilation {
        kernel: LogicalKernelId,
        source: ComputeError,
    },
}

impl fmt::Display for PlanPreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::NonDenseKernelId { position, found } =>
            {
                write!(
                    formatter,
                    "kernel at position {position} declares id {}; ids must equal their position",
                    found.get()
                )
            },
            Self::UnknownKernelId {
                dispatch_index,
                kernel,
            } =>
            {
                write!(
                    formatter,
                    "dispatch {dispatch_index} names kernel {}, which the plan does not contain",
                    kernel.get()
                )
            },
            Self::UnknownBindingId {
                dispatch_index,
                binding,
            } =>
            {
                write!(
                    formatter,
                    "dispatch {dispatch_index} names external binding {}, which the plan does not contain",
                    binding.get()
                )
            },
            Self::BindingKindMismatch {
                dispatch_index,
                binding,
            } =>
            {
                write!(
                    formatter,
                    "dispatch {dispatch_index} uses external binding {} with the wrong kind",
                    binding.get()
                )
            },
            Self::NonDenseBufferSlot { slot } =>
            {
                write!(
                    formatter,
                    "buffer slot {} is never referenced, so slots are not dense",
                    slot.get()
                )
            },
            Self::ArgumentCountMismatch {
                dispatch_index,
                expected,
                actual,
            } =>
            {
                write!(
                    formatter,
                    "dispatch {dispatch_index} needs {expected} argument(s) but carries {actual}"
                )
            },
            Self::ArgumentIndexMismatch {
                dispatch_index,
                position,
                found,
            } =>
            {
                write!(
                    formatter,
                    "dispatch {dispatch_index} argument at position {position} declares index {found}"
                )
            },
            Self::MissingWriteArgument { dispatch_index } =>
            {
                write!(formatter, "dispatch {dispatch_index} has no write argument")
            },
            Self::MultipleWriteArguments { dispatch_index } =>
            {
                write!(
                    formatter,
                    "dispatch {dispatch_index} has more than one write argument"
                )
            },
            Self::WriteArgumentNotLast {
                dispatch_index,
                position,
            } =>
            {
                write!(
                    formatter,
                    "dispatch {dispatch_index} writes at position {position}, not last"
                )
            },
            Self::WriteToExternalValue {
                dispatch_index,
                binding,
            } =>
            {
                write!(
                    formatter,
                    "dispatch {dispatch_index} writes to external binding {}, which is immutable",
                    binding.get()
                )
            },
            Self::ArgumentTypeMismatch {
                dispatch_index,
                position,
            } =>
            {
                write!(
                    formatter,
                    "dispatch {dispatch_index} argument {position} disagrees with its kernel signature"
                )
            },
            Self::InconsistentSlotType { slot } =>
            {
                write!(
                    formatter,
                    "buffer slot {} is used with more than one tensor type",
                    slot.get()
                )
            },
            Self::InconsistentExternalType { binding } =>
            {
                write!(
                    formatter,
                    "external binding {} is used with more than one tensor type",
                    binding.get()
                )
            },
            Self::UndeterminedValueType { node } =>
            {
                write!(
                    formatter,
                    "node {} has no tensor type anywhere in the plan",
                    node.get()
                )
            },
            Self::UnsupportedDType { node } =>
            {
                write!(
                    formatter,
                    "node {} uses an element type this runtime cannot execute; only F32 is supported",
                    node.get()
                )
            },
            Self::BackendDTypeUnsupported =>
            {
                formatter.write_str("the backend does not advertise support for F32")
            },
            Self::BackendWorkgroupTooSmall { max_workgroup_size } =>
            {
                write!(
                    formatter,
                    "the backend advertises a maximum workgroup of {max_workgroup_size:?}, \
                     smaller than the canonical [1, 1, 1] launch"
                )
            },
            Self::DispatchExtentMismatch {
                dispatch_index,
                expected,
                actual,
            } =>
            {
                write!(
                    formatter,
                    "dispatch {dispatch_index} declares {actual} element(s) but its result holds {expected}"
                )
            },
            Self::ReadBeforeDefinition {
                dispatch_index,
                slot,
            } =>
            {
                write!(
                    formatter,
                    "dispatch {dispatch_index} reads buffer slot {} before any dispatch defines it",
                    slot.get()
                )
            },
            Self::UnsupportedArgumentAccess {
                dispatch_index,
                position,
            } =>
            {
                write!(
                    formatter,
                    "dispatch {dispatch_index} argument {position} declares an access kind this runtime does not handle"
                )
            },
            Self::UnsupportedArgumentSource {
                dispatch_index,
                position,
            } =>
            {
                write!(
                    formatter,
                    "dispatch {dispatch_index} argument {position} uses a source kind this runtime does not handle"
                )
            },
            Self::UnknownOutputSource { output_index, node } =>
            {
                write!(
                    formatter,
                    "output {output_index} (node {}) names a source the plan does not contain",
                    node.get()
                )
            },
            Self::SizeOverflow { elements } =>
            {
                write!(
                    formatter,
                    "a buffer of {elements} element(s) overflows usize when sized in bytes"
                )
            },
            Self::BufferExceedsBackendLimit { bytes, limit } =>
            {
                write!(
                    formatter,
                    "a buffer of {bytes} byte(s) exceeds the backend's {limit}-byte maximum"
                )
            },
            Self::DeterministicMathUnavailable { kernel, opcode } =>
            {
                write!(
                    formatter,
                    "kernel {} uses {opcode:?}, which this runtime does not execute: no \
                     bit-reproducible implementation is available",
                    kernel.get()
                )
            },
            Self::KernelGeneration { source } =>
            {
                write!(formatter, "reference artefact generation failed: {source}")
            },
            Self::KernelArtifactMismatch { expected, actual } =>
            {
                write!(
                    formatter,
                    "generation produced {actual} artefact(s) for {expected} kernel(s)"
                )
            },
            Self::KernelModule { kernel, source } =>
            {
                write!(
                    formatter,
                    "kernel {} could not become a kernel module: {source}",
                    kernel.get()
                )
            },
            Self::KernelCompilation { kernel, source } =>
            {
                write!(
                    formatter,
                    "the backend rejected kernel {}: {source}",
                    kernel.get()
                )
            },
        }
    }
}

impl core::error::Error for PlanPreparationError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self
        {
            Self::KernelGeneration { source } | Self::KernelModule { source, .. } => Some(source),
            Self::KernelCompilation { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// A failure while running a prepared plan.
///
/// Every dispatch-scoped variant carries the dispatch index, the canonical
/// `NodeId` it produces and the `LogicalKernelId` it runs, so a failure can be
/// located in the plan without re-deriving anything.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlanExecutionError {
    /// The plan needs a value for this binding and none was supplied.
    MissingExternalValue { binding: LogicalBindingId },
    /// A value was supplied for a binding the plan does not have.
    UnexpectedExternalValue { binding: LogicalBindingId },
    /// The same binding was supplied more than once.
    DuplicateExternalValue { binding: LogicalBindingId },
    /// A supplied value does not hold the number of elements the plan expects.
    ExternalValueLengthMismatch {
        binding: LogicalBindingId,
        expected: usize,
        actual: usize,
    },
    /// A buffer's byte size overflows `usize`.
    ///
    /// Defensive: preparation already sized every buffer with a checked
    /// multiplication.
    SizeOverflow { elements: usize },
    /// The backend could not allocate a buffer.
    BufferAllocation { bytes: usize, source: ComputeError },
    /// The backend could not write an external value into its buffer.
    BufferWrite {
        binding: LogicalBindingId,
        source: ComputeError,
    },
    /// The backend could not create the execution stream.
    StreamCreation { source: ComputeError },
    /// A physical buffer expected by a dispatch is absent.
    ///
    /// Defensive: preparation resolved every argument to a buffer index, and
    /// execution allocates exactly those buffers.
    MissingPhysicalBuffer { dispatch_index: usize },
    /// A compiled kernel expected by a dispatch is absent.
    ///
    /// Defensive, for the same reason as [`Self::MissingPhysicalBuffer`].
    MissingCompiledKernel {
        dispatch_index: usize,
        kernel: LogicalKernelId,
    },
    /// The backend rejected a dispatch.
    KernelLaunch {
        dispatch_index: usize,
        node: NodeId,
        kernel: LogicalKernelId,
        source: ComputeError,
    },
    /// Waiting on a dispatch's completion event failed.
    EventWait {
        dispatch_index: usize,
        node: NodeId,
        kernel: LogicalKernelId,
        source: ComputeError,
    },
    /// Synchronising the stream after the last dispatch failed.
    Synchronization { source: ComputeError },
    /// Reading an output back from its buffer failed.
    BufferRead {
        output_index: usize,
        node: NodeId,
        source: ComputeError,
    },
    /// An output's byte range is not a whole number of `f32` values.
    ///
    /// Defensive: every buffer is sized as `elements * 4`.
    MalformedOutputBytes {
        output_index: usize,
        length_bytes: usize,
    },
}

impl fmt::Display for PlanExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::MissingExternalValue { binding } =>
            {
                write!(
                    formatter,
                    "no value supplied for external binding {}",
                    binding.get()
                )
            },
            Self::UnexpectedExternalValue { binding } =>
            {
                write!(
                    formatter,
                    "a value was supplied for binding {}, which the plan does not have",
                    binding.get()
                )
            },
            Self::DuplicateExternalValue { binding } =>
            {
                write!(
                    formatter,
                    "binding {} was supplied more than once",
                    binding.get()
                )
            },
            Self::ExternalValueLengthMismatch {
                binding,
                expected,
                actual,
            } =>
            {
                write!(
                    formatter,
                    "binding {} expects {expected} element(s) but received {actual}",
                    binding.get()
                )
            },
            Self::SizeOverflow { elements } =>
            {
                write!(
                    formatter,
                    "a buffer of {elements} element(s) overflows usize when sized in bytes"
                )
            },
            Self::BufferAllocation { bytes, source } =>
            {
                write!(
                    formatter,
                    "allocating a {bytes}-byte buffer failed: {source}"
                )
            },
            Self::BufferWrite { binding, source } =>
            {
                write!(
                    formatter,
                    "writing external binding {} failed: {source}",
                    binding.get()
                )
            },
            Self::StreamCreation { source } =>
            {
                write!(formatter, "creating the execution stream failed: {source}")
            },
            Self::MissingPhysicalBuffer { dispatch_index } =>
            {
                write!(
                    formatter,
                    "dispatch {dispatch_index} references a physical buffer that was not allocated"
                )
            },
            Self::MissingCompiledKernel {
                dispatch_index,
                kernel,
            } =>
            {
                write!(
                    formatter,
                    "dispatch {dispatch_index} references uncompiled kernel {}",
                    kernel.get()
                )
            },
            Self::KernelLaunch {
                dispatch_index,
                node,
                kernel,
                source,
            } =>
            {
                write!(
                    formatter,
                    "dispatch {dispatch_index} (node {}, kernel {}) failed to launch: {source}",
                    node.get(),
                    kernel.get()
                )
            },
            Self::EventWait {
                dispatch_index,
                node,
                kernel,
                source,
            } =>
            {
                write!(
                    formatter,
                    "waiting on dispatch {dispatch_index} (node {}, kernel {}) failed: {source}",
                    node.get(),
                    kernel.get()
                )
            },
            Self::Synchronization { source } =>
            {
                write!(formatter, "stream synchronization failed: {source}")
            },
            Self::BufferRead {
                output_index,
                node,
                source,
            } =>
            {
                write!(
                    formatter,
                    "reading output {output_index} (node {}) failed: {source}",
                    node.get()
                )
            },
            Self::MalformedOutputBytes {
                output_index,
                length_bytes,
            } =>
            {
                write!(
                    formatter,
                    "output {output_index} holds {length_bytes} byte(s), not a whole number of f32 values"
                )
            },
        }
    }
}

impl core::error::Error for PlanExecutionError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self
        {
            Self::BufferAllocation { source, .. }
            | Self::BufferWrite { source, .. }
            | Self::StreamCreation { source }
            | Self::KernelLaunch { source, .. }
            | Self::EventWait { source, .. }
            | Self::Synchronization { source }
            | Self::BufferRead { source, .. } => Some(source),
            _ => None,
        }
    }
}
