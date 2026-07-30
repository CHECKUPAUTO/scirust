//! Typed failures of the Reference plan runtime and of the graph session.
//!
//! Each layer splits preparation from execution, so a caller can tell "this can
//! never run here" from "this run did not work out":
//!
//! * [`PlanPreparationError`] / [`PlanExecutionError`] for a `LoweredPlan`;
//! * [`GraphSessionPreparationError`] / [`GraphSessionExecutionError`] for a
//!   canonical `Graph`.
//!
//! Every encapsulated failure — a backend [`ComputeError`], a compiler, lowerer
//! or plan-runtime error — is kept whole and reachable through
//! [`core::error::Error::source`] rather than flattened into a message, and no
//! variant anywhere is only a free-form string.

use core::fmt;

use scirust_compute::ComputeError;
use scirust_tensor_compile::{
    BufferSlot, CompileError, ExternalValueKind, LogicalBindingId, LogicalKernelId, LoweringError,
};
use scirust_tensor_ir::{ConstantId, DType, NodeId, TensorType};
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
    ///
    /// A caller holding the canonical `Graph` can supply the missing type to
    /// `ReferencePlanRuntime::prepare_with_external_types`.
    UndeterminedValueType { node: NodeId },
    /// An external type hint names a node that is not an external binding of the
    /// plan.
    ///
    /// Hints complete existing metadata; they never add a binding.
    UnknownExternalTypeNode { node: NodeId },
    /// The same node is given an external type hint more than once.
    DuplicateExternalType { node: NodeId },
    /// An external type hint disagrees with the type the plan itself states for
    /// that value through a kernel argument.
    ///
    /// `expected` is the hint, `actual` the occurrence found in the plan. A hint
    /// may fill a gap; it may never overrule the plan.
    ExternalTypeContradiction {
        binding: LogicalBindingId,
        node: NodeId,
        expected: TensorType,
        actual: TensorType,
    },
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
            Self::UnknownExternalTypeNode { node } =>
            {
                write!(
                    formatter,
                    "external type supplied for node {}, which the plan does not bind",
                    node.get()
                )
            },
            Self::DuplicateExternalType { node } =>
            {
                write!(
                    formatter,
                    "external type supplied more than once for node {}",
                    node.get()
                )
            },
            Self::ExternalTypeContradiction {
                binding,
                node,
                expected,
                actual,
            } =>
            {
                write!(
                    formatter,
                    "external type {expected:?} supplied for node {} contradicts {actual:?}, \
                     the type binding {} carries in the plan",
                    node.get(),
                    binding.get()
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

// ---------------------------------------------------------------------------
// Graph session
// ---------------------------------------------------------------------------

/// A failure while turning a canonical `Graph` into a reusable session.
///
/// The three encapsulating variants keep the underlying compiler, lowerer and
/// plan-runtime errors intact and reachable through [`core::error::Error::source`]
/// rather than flattening them into a message.
///
/// Several variants guard invariants the canonical pipeline already establishes
/// — a binding always names an `Input` or `Constant` node of the graph, and the
/// plan's outputs are the graph's outputs, copied verbatim. They are checked
/// anyway rather than assumed, and each says so in its own documentation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GraphSessionPreparationError {
    /// `CanonicalCompiler` rejected the graph.
    GraphCompilation { source: CompileError },
    /// `KernelLowerer` rejected the execution plan.
    KernelLowering { source: LoweringError },
    /// The Reference plan runtime rejected the lowered plan.
    PlanPreparation { source: PlanPreparationError },
    /// A binding names a node the graph does not declare as an external value:
    /// either absent from the graph, or carrying an operation that is neither
    /// `Input` nor `Constant`.
    ///
    /// Defensive: the binding table is derived from the graph's own memory plan,
    /// which classifies a value as external exactly when its operation is
    /// `Input` or `Constant`.
    UnknownExternalNode { node: NodeId },
    /// The graph and the plan disagree on whether a value is an input or a
    /// constant.
    ///
    /// `expected` is the graph's classification, `actual` the plan's.
    ///
    /// Defensive, for the same reason as [`Self::UnknownExternalNode`].
    ExternalKindMismatch {
        node: NodeId,
        expected: ExternalValueKind,
        actual: ExternalValueKind,
    },
    /// A binding declares an external kind this session does not handle.
    ///
    /// `ExternalValueKind` is `#[non_exhaustive]`: a variant added to it later is
    /// rejected here instead of being silently treated as an input or a
    /// constant.
    UnsupportedExternalKind { node: NodeId },
    /// A graph input uses an element type this session cannot supply.
    UnsupportedInputDType { node: NodeId, dtype: DType },
    /// A graph constant uses an element type this session cannot supply.
    UnsupportedConstantDType { node: NodeId, dtype: DType },
    /// An input's element count overflows `usize`.
    InputSizeOverflow { node: NodeId },
    /// A constant's element count overflows `usize`.
    ConstantSizeOverflow { node: NodeId },
    /// A constant survives dead-code elimination but no payload was supplied for
    /// its `ConstantId`.
    MissingConstantPayload { node: NodeId, constant: ConstantId },
    /// A payload was supplied for a `ConstantId` no node of the graph
    /// references.
    ///
    /// A payload for a constant the graph *does* declare but dead-code
    /// elimination removed is accepted and ignored; this variant catches a
    /// mistyped identifier, which silence would hide.
    UnexpectedConstantPayload { constant: ConstantId },
    /// Two payloads were supplied for the same `ConstantId`.
    DuplicateConstantPayload { constant: ConstantId },
    /// A constant's payload does not hold the number of elements its node
    /// declares.
    ///
    /// One payload serves every surviving node sharing the `ConstantId`, so it
    /// must satisfy all of them.
    ConstantLengthMismatch {
        node: NodeId,
        constant: ConstantId,
        expected: usize,
        actual: usize,
    },
    /// The prepared plan does not expose the number of external values the
    /// session resolved from the binding table.
    ///
    /// Defensive: preparation supplies the plan's own bindings, and hints never
    /// add or remove one.
    PreparedExternalCountMismatch { expected: usize, actual: usize },
    /// The prepared plan's external value at this position does not match the
    /// binding the session resolved for it.
    ///
    /// Defensive, for the same reason as
    /// [`Self::PreparedExternalCountMismatch`].
    PreparedExternalMismatch {
        binding: LogicalBindingId,
        node: NodeId,
    },
    /// A declared output names a node absent from the graph.
    ///
    /// Defensive: `Graph::validate`, which `CanonicalCompiler` runs first,
    /// rejects an output outside the node range.
    UnknownOutputNode { node: NodeId },
    /// The graph and the prepared plan declare different numbers of outputs.
    ///
    /// Defensive: the plan's outputs are the graph's, copied verbatim.
    OutputCountMismatch { graph: usize, plan: usize },
    /// The graph and the prepared plan disagree on an output's node.
    ///
    /// Defensive, for the same reason as [`Self::OutputCountMismatch`].
    OutputNodeMismatch {
        index: usize,
        graph: NodeId,
        plan: NodeId,
    },
    /// The graph and the prepared plan disagree on an output's tensor type.
    ///
    /// Defensive: both descend from the same `Node::output`.
    OutputTypeMismatch {
        index: usize,
        node: NodeId,
        graph: TensorType,
        plan: TensorType,
    },
}

impl fmt::Display for GraphSessionPreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::GraphCompilation { source } =>
            {
                write!(formatter, "canonical compilation failed: {source}")
            },
            Self::KernelLowering { source } =>
            {
                write!(formatter, "kernel lowering failed: {source}")
            },
            Self::PlanPreparation { source } =>
            {
                write!(formatter, "reference plan preparation failed: {source}")
            },
            Self::UnknownExternalNode { node } =>
            {
                write!(
                    formatter,
                    "binding names node {}, which the graph does not declare as an external value",
                    node.get()
                )
            },
            Self::ExternalKindMismatch {
                node,
                expected,
                actual,
            } =>
            {
                write!(
                    formatter,
                    "node {} is an external {expected:?} in the graph but {actual:?} in the plan",
                    node.get()
                )
            },
            Self::UnsupportedExternalKind { node } =>
            {
                write!(
                    formatter,
                    "node {} declares an external kind this session does not handle",
                    node.get()
                )
            },
            Self::UnsupportedInputDType { node, dtype } =>
            {
                write!(
                    formatter,
                    "input node {} uses {dtype:?}; this session supplies F32 only",
                    node.get()
                )
            },
            Self::UnsupportedConstantDType { node, dtype } =>
            {
                write!(
                    formatter,
                    "constant node {} uses {dtype:?}; this session supplies F32 only",
                    node.get()
                )
            },
            Self::InputSizeOverflow { node } =>
            {
                write!(
                    formatter,
                    "element count of input node {} overflows usize",
                    node.get()
                )
            },
            Self::ConstantSizeOverflow { node } =>
            {
                write!(
                    formatter,
                    "element count of constant node {} overflows usize",
                    node.get()
                )
            },
            Self::MissingConstantPayload { node, constant } =>
            {
                write!(
                    formatter,
                    "constant node {} needs a payload for constant {}, and none was supplied",
                    node.get(),
                    constant.get()
                )
            },
            Self::UnexpectedConstantPayload { constant } =>
            {
                write!(
                    formatter,
                    "a payload was supplied for constant {}, which no node of the graph references",
                    constant.get()
                )
            },
            Self::DuplicateConstantPayload { constant } =>
            {
                write!(
                    formatter,
                    "constant {} was supplied more than once",
                    constant.get()
                )
            },
            Self::ConstantLengthMismatch {
                node,
                constant,
                expected,
                actual,
            } =>
            {
                write!(
                    formatter,
                    "constant node {} expects {expected} element(s) but constant {} supplies \
                     {actual}",
                    node.get(),
                    constant.get()
                )
            },
            Self::PreparedExternalCountMismatch { expected, actual } =>
            {
                write!(
                    formatter,
                    "the prepared plan exposes {actual} external value(s) for {expected} binding(s)"
                )
            },
            Self::PreparedExternalMismatch { binding, node } =>
            {
                write!(
                    formatter,
                    "the prepared plan's external value {} does not describe node {}",
                    binding.get(),
                    node.get()
                )
            },
            Self::UnknownOutputNode { node } =>
            {
                write!(
                    formatter,
                    "declared output {} is absent from the graph",
                    node.get()
                )
            },
            Self::OutputCountMismatch { graph, plan } =>
            {
                write!(
                    formatter,
                    "the graph declares {graph} output(s) and the prepared plan {plan}"
                )
            },
            Self::OutputNodeMismatch { index, graph, plan } =>
            {
                write!(
                    formatter,
                    "output {index} is node {} in the graph and node {} in the prepared plan",
                    graph.get(),
                    plan.get()
                )
            },
            Self::OutputTypeMismatch {
                index,
                node,
                graph,
                plan,
            } =>
            {
                write!(
                    formatter,
                    "output {index} (node {}) is {graph:?} in the graph and {plan:?} in the \
                     prepared plan",
                    node.get()
                )
            },
        }
    }
}

impl core::error::Error for GraphSessionPreparationError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self
        {
            Self::GraphCompilation { source } => Some(source),
            Self::KernelLowering { source } => Some(source),
            Self::PlanPreparation { source } => Some(source),
            _ => None,
        }
    }
}

/// A failure while running a prepared session.
///
/// Every input fault is named by the caller's own vocabulary — the graph
/// `NodeId` — never by the internal `LogicalBindingId` the session resolves it
/// to. All four are raised before any backend call.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GraphSessionExecutionError {
    /// The session needs a value for this input node and none was supplied.
    MissingInput { node: NodeId },
    /// A value was supplied for a node that is not a required input of this
    /// session.
    ///
    /// That covers a node the graph does not declare as an input, a constant —
    /// constants are never supplied per execution — and an input dead-code
    /// elimination removed from the prepared plan.
    UnexpectedInput { node: NodeId },
    /// The same input node was supplied more than once.
    DuplicateInput { node: NodeId },
    /// A supplied value does not hold the number of elements the input's tensor
    /// type declares.
    InputLengthMismatch {
        node: NodeId,
        expected: usize,
        actual: usize,
    },
    /// A constant binding of the session has no stored payload.
    ///
    /// Defensive: preparation stores exactly one payload per surviving constant
    /// binding and never mutates the table afterwards.
    UnresolvedConstantPayload { binding: LogicalBindingId },
    /// The prepared plan failed to run.
    PlanExecution { source: PlanExecutionError },
}

impl fmt::Display for GraphSessionExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::MissingInput { node } =>
            {
                write!(formatter, "input node {} was not supplied", node.get())
            },
            Self::UnexpectedInput { node } =>
            {
                write!(
                    formatter,
                    "node {} is not a required input of this session",
                    node.get()
                )
            },
            Self::DuplicateInput { node } =>
            {
                write!(
                    formatter,
                    "input node {} was supplied more than once",
                    node.get()
                )
            },
            Self::InputLengthMismatch {
                node,
                expected,
                actual,
            } =>
            {
                write!(
                    formatter,
                    "input node {} expects {expected} element(s) but received {actual}",
                    node.get()
                )
            },
            Self::UnresolvedConstantPayload { binding } =>
            {
                write!(
                    formatter,
                    "constant binding {} has no stored payload",
                    binding.get()
                )
            },
            Self::PlanExecution { source } =>
            {
                write!(formatter, "prepared plan execution failed: {source}")
            },
        }
    }
}

impl core::error::Error for GraphSessionExecutionError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self
        {
            Self::PlanExecution { source } => Some(source),
            _ => None,
        }
    }
}
