//! Logical kernel-lowering contract for canonical execution plans.
//!
//! This module is private; every type below is re-exported at the crate root and
//! documented there. The crate-level documentation is the reference for the
//! contract itself: the four planes of the compilation pipeline, the difference
//! between a logical binding and a physical buffer binding, why a lowered plan
//! holds no compiled kernel, the deterministic invariants, and the operations
//! this phase supports and rejects.
//!
//! Notes for maintainers of this module:
//!
//! * Nothing here may reference a backend, a physical buffer, a device, a
//!   stream, an event, a compiled kernel or launch geometry. The crate depends
//!   on neither `scirust-gpu` nor `scirust-core`, and that boundary is what
//!   keeps the lowering reusable by every future generator.
//! * `NodeId` values are *not* positions: dead-code elimination in
//!   [`crate::CanonicalCompiler`] keeps original identifiers while dropping
//!   nodes, so every lookup goes through an ordered map, never through indexing
//!   by `NodeId::get`.
//! * The public lowering path contains no `unwrap`, `expect`, `panic!` or
//!   unchecked indexing. Every failure is a typed [`LoweringError`].

use core::fmt;
use std::collections::BTreeMap;

use scirust_tensor_ir::{NodeId, Operation, Scalar, Shape, TensorType};

use crate::{BufferSlot, ExecutionPlan, Instruction, ValueStorage};

/// Stable identifier of one entry in the external binding table.
///
/// The value is the position of the entry in the table supplied to
/// [`KernelLowerer::lower`], and nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LogicalBindingId(u32);

impl LogicalBindingId {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Stable identifier of one logical kernel signature.
///
/// The value is the position of the kernel in [`LoweredPlan::kernels`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LogicalKernelId(u32);

impl LogicalKernelId {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Identifier space that overflowed its `u32` capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum IdentifierSpace {
    ExternalBinding,
    LogicalKernel,
    KernelArgument,
}

/// Class of external value the runtime is expected to supply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum ExternalValueKind {
    Input,
    Constant,
}

/// One externally supplied value of the plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternalBinding {
    pub node: NodeId,
    pub kind: ExternalValueKind,
}

/// Ordered table of the values a runtime must supply before dispatching.
///
/// The position of an entry *is* its [`LogicalBindingId`], so the caller
/// controls the external calling order by controlling this order.
///
/// [`ExternalBindings::from_entries`] performs no validation on purpose:
/// completeness, uniqueness and kind agreement are all checked in one place, by
/// [`KernelLowerer::lower`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExternalBindings {
    entries: Vec<ExternalBinding>,
}

impl ExternalBindings {
    /// Deterministic default table: every external value of the plan, in the
    /// canonical order of the retained instructions.
    pub fn derive(plan: &ExecutionPlan) -> Self {
        let entries = plan
            .memory_plan()
            .allocations()
            .iter()
            .filter_map(|allocation| {
                external_kind(allocation.storage).map(|kind| ExternalBinding {
                    node: allocation.node,
                    kind,
                })
            })
            .collect();

        Self { entries }
    }

    /// Table with a caller-chosen order. Validated by [`KernelLowerer::lower`].
    pub fn from_entries(entries: Vec<ExternalBinding>) -> Self {
        Self { entries }
    }

    pub fn entries(&self) -> &[ExternalBinding] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Element-wise kernel over one operand.
///
/// The `Scale` factor is part of the signature, not a runtime argument: it is
/// bit-exact ([`Scalar`] stores raw bits), so deduplication stays deterministic
/// and a generator can fold it as a literal without any uniform-buffer ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum UnaryKernel {
    Relu,
    Exp,
    Log,
    Scale { factor: Scalar },
}

/// Element-wise kernel over two operands of identical type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum BinaryKernel {
    Add,
    Sub,
    Mul,
    Div,
}

/// Closed set of logical kernel shapes a generator has to handle.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum KernelFamily {
    ElementwiseUnary(UnaryKernel),
    ElementwiseBinary(BinaryKernel),
    /// Copy preserving the linear element order into a differently shaped
    /// destination — the lowering of `Reshape`.
    ///
    /// This phase has no aliasing contract in [`crate::MemoryPlan`]: the planner
    /// reserves a distinct [`BufferSlot`] for the reshaped value, so a free
    /// "view" would contradict the memory plan. `ShapeCopy` therefore writes
    /// into that distinct slot, honestly. Turning it into a zero-copy view later
    /// requires a real aliasing contract in the memory plan, and must not change
    /// the observable semantics.
    ShapeCopy,
    /// Rank-preserving index permutation — the lowering of `Transpose`.
    ///
    /// Convention, validated at lowering time:
    /// `output.shape[i] == input.shape[permutation[i]]`. The permutation lists,
    /// in output-axis order, the corresponding input axes, as with NumPy.
    Permute {
        permutation: Vec<usize>,
    },
}

/// One unit of code to be generated, fully specialised.
///
/// Operand and result types are part of the signature, so a generated kernel is
/// self-contained and needs no shape metadata at dispatch time. The cost is one
/// kernel per distinct type combination; shape-generic kernels would require a
/// metadata argument ABI that this phase does not define.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalKernel {
    id: LogicalKernelId,
    family: KernelFamily,
    operands: Vec<TensorType>,
    result: TensorType,
}

impl LogicalKernel {
    pub const fn id(&self) -> LogicalKernelId {
        self.id
    }

    pub const fn family(&self) -> &KernelFamily {
        &self.family
    }

    pub fn operands(&self) -> &[TensorType] {
        &self.operands
    }

    pub const fn result(&self) -> &TensorType {
        &self.result
    }
}

/// Access rights of one logical argument, stated explicitly.
///
/// `ReadWrite` is deliberately absent. [`crate::MemoryPlan`] retires a slot only
/// once its last use is strictly before the current step, so an operand read at
/// a step can never share its slot with that step's result: no supported family
/// needs in-place access, and offering the variant would misstate the contract.
///
/// A backend may still require wider physical rights than the logical contract
/// grants — WGSL storage buffers reject write-only bindings, for instance. That
/// widening is a generator and runtime decision, not a property of this plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum KernelArgumentAccess {
    Read,
    Write,
}

/// Value that fills one logical argument.
///
/// No pointer, no backend buffer, no device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum KernelArgumentSource {
    ExternalInput(LogicalBindingId),
    ExternalConstant(LogicalBindingId),
    Buffer(BufferSlot),
}

/// One logical argument of one dispatch.
///
/// `index` is the stable logical slot a runtime will later map onto a physical
/// binding slot. It equals the argument's position in
/// [`LoweredInstruction::arguments`], and the lowerer verifies that.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelArgument {
    pub index: u32,
    pub source: KernelArgumentSource,
    pub access: KernelArgumentAccess,
    pub tensor_type: TensorType,
}

/// Logical iteration space of one dispatch.
///
/// Not a launch configuration: no grid, no block, no shared memory. `elements`
/// is computed with the checked shape API, never by unchecked multiplication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalDispatch {
    pub extent: Shape,
    pub elements: usize,
}

/// One lowered dispatch, keeping its canonical identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoweredInstruction {
    /// Canonical node whose value this dispatch produces.
    pub node: NodeId,
    /// Canonical nodes covered by this dispatch.
    ///
    /// Always exactly one in this phase. The field exists so a later fusion pass
    /// can describe one kernel covering several canonical nodes without
    /// reshaping this contract.
    pub sources: Vec<NodeId>,
    pub kernel: LogicalKernelId,
    pub arguments: Vec<KernelArgument>,
    pub dispatch: LogicalDispatch,
}

/// One plan output and the logical value holding it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoweredOutput {
    pub node: NodeId,
    pub source: KernelArgumentSource,
}

/// Backend-neutral logical lowering of an [`ExecutionPlan`].
///
/// Contains no backend, no physical buffer, no stream, no event, no compiled
/// kernel and no device. It cannot allocate and it cannot launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoweredPlan {
    kernels: Vec<LogicalKernel>,
    instructions: Vec<LoweredInstruction>,
    bindings: Vec<ExternalBinding>,
    outputs: Vec<LoweredOutput>,
}

impl LoweredPlan {
    /// Deduplicated kernel signatures, indexed by [`LogicalKernelId`].
    pub fn kernels(&self) -> &[LogicalKernel] {
        &self.kernels
    }

    /// Dispatches, in canonical instruction order.
    pub fn instructions(&self) -> &[LoweredInstruction] {
        &self.instructions
    }

    /// External values to supply, indexed by [`LogicalBindingId`].
    pub fn bindings(&self) -> &[ExternalBinding] {
        &self.bindings
    }

    /// Plan outputs, in the order declared by the canonical plan.
    pub fn outputs(&self) -> &[LoweredOutput] {
        &self.outputs
    }

    pub fn kernel(&self, id: LogicalKernelId) -> Option<&LogicalKernel> {
        usize::try_from(id.get())
            .ok()
            .and_then(|index| self.kernels.get(index))
    }

    pub fn binding(&self, id: LogicalBindingId) -> Option<ExternalBinding> {
        usize::try_from(id.get())
            .ok()
            .and_then(|index| self.bindings.get(index))
            .copied()
    }

    /// Dispatch producing the value of `node`, if that node is computed.
    pub fn instruction_for(&self, node: NodeId) -> Option<&LoweredInstruction> {
        self.instructions
            .iter()
            .find(|instruction| instruction.node == node)
    }
}

/// Typed lowering failures.
///
/// `MissingBufferSlot`, `ArityMismatch`, `UnloweredOutput`,
/// `ArgumentIndexMismatch` and `IdentifierOverflow` guard invariants that
/// [`crate::CanonicalCompiler`] already establishes today, so they are not
/// reachable through the current public API. They exist so that a future change
/// to plan construction fails loudly with a typed error instead of panicking or
/// silently producing a wrong plan, and they are honestly not covered by tests.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum LoweringError {
    /// The operation has no logical kernel in this phase.
    UnsupportedOperation { node: NodeId, operation: Operation },
    /// An external value of the plan is absent from the binding table.
    MissingExternalBinding { node: NodeId },
    /// The binding table names the same node twice.
    DuplicateExternalBinding { node: NodeId },
    /// The binding table names a node that is not an external value.
    ExtraneousExternalBinding { node: NodeId },
    /// The declared binding kind disagrees with the plan's storage class.
    ExternalBindingKindMismatch {
        node: NodeId,
        expected: ExternalValueKind,
        actual: ExternalValueKind,
    },
    /// A reference names a node absent from the plan.
    UnknownNode { node: NodeId },
    /// A plan value has no storage assignment.
    MissingBufferSlot { node: NodeId },
    /// The instruction operand count disagrees with the operation arity.
    ArityMismatch {
        node: NodeId,
        expected: usize,
        actual: usize,
    },
    /// A declared output has no lowered value.
    UnloweredOutput { node: NodeId },
    /// An operand type disagrees with the operation's requirement.
    OperandTypeMismatch {
        node: NodeId,
        expected: TensorType,
        actual: TensorType,
    },
    /// Reshape dtype, element count or declared target shape is inconsistent.
    InvalidReshape { node: NodeId },
    /// Transpose permutation or resulting shape is inconsistent.
    InvalidPermutation { node: NodeId },
    /// The dispatch element count overflows `usize`.
    DispatchExtentOverflow { node: NodeId },
    /// An argument's logical index does not equal its position.
    ArgumentIndexMismatch {
        node: NodeId,
        expected: u32,
        actual: u32,
    },
    /// A stable identifier space exceeded its `u32` capacity.
    IdentifierOverflow {
        space: IdentifierSpace,
        node: NodeId,
    },
}

impl fmt::Display for LoweringError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::UnsupportedOperation { node, operation } =>
            {
                write!(
                    formatter,
                    "node {} carries operation {operation:?}, which has no logical kernel in this lowering phase",
                    node.get()
                )
            },
            Self::MissingExternalBinding { node } =>
            {
                write!(
                    formatter,
                    "external value of node {} is not bound",
                    node.get()
                )
            },
            Self::DuplicateExternalBinding { node } =>
            {
                write!(formatter, "node {} is bound more than once", node.get())
            },
            Self::ExtraneousExternalBinding { node } =>
            {
                write!(
                    formatter,
                    "node {} is bound but is not an external value of the plan",
                    node.get()
                )
            },
            Self::ExternalBindingKindMismatch {
                node,
                expected,
                actual,
            } =>
            {
                write!(
                    formatter,
                    "node {} is an external {expected:?} but is bound as {actual:?}",
                    node.get()
                )
            },
            Self::UnknownNode { node } =>
            {
                write!(
                    formatter,
                    "node {} is absent from the execution plan",
                    node.get()
                )
            },
            Self::MissingBufferSlot { node } =>
            {
                write!(
                    formatter,
                    "node {} has no storage assignment in the memory plan",
                    node.get()
                )
            },
            Self::ArityMismatch {
                node,
                expected,
                actual,
            } =>
            {
                write!(
                    formatter,
                    "node {} expects {expected} operands but carries {actual}",
                    node.get()
                )
            },
            Self::UnloweredOutput { node } =>
            {
                write!(
                    formatter,
                    "declared output {} has no lowered value",
                    node.get()
                )
            },
            Self::OperandTypeMismatch {
                node,
                expected,
                actual,
            } =>
            {
                write!(
                    formatter,
                    "node {} expects operand type {expected:?} but received {actual:?}",
                    node.get()
                )
            },
            Self::InvalidReshape { node } =>
            {
                write!(
                    formatter,
                    "node {} declares an inconsistent reshape",
                    node.get()
                )
            },
            Self::InvalidPermutation { node } =>
            {
                write!(
                    formatter,
                    "node {} declares an inconsistent transpose permutation",
                    node.get()
                )
            },
            Self::DispatchExtentOverflow { node } =>
            {
                write!(
                    formatter,
                    "dispatch extent of node {} overflows usize",
                    node.get()
                )
            },
            Self::ArgumentIndexMismatch {
                node,
                expected,
                actual,
            } =>
            {
                write!(
                    formatter,
                    "node {} argument index {actual} does not match its position {expected}",
                    node.get()
                )
            },
            Self::IdentifierOverflow { space, node } =>
            {
                write!(
                    formatter,
                    "{space:?} identifier space overflowed u32 while lowering node {}",
                    node.get()
                )
            },
        }
    }
}

impl std::error::Error for LoweringError {}

/// Storage class and type of one canonical value.
#[derive(Debug, Clone)]
struct ValueInfo {
    storage: ValueStorage,
    tensor_type: TensorType,
}

/// External class of a storage assignment, if any.
const fn external_kind(storage: ValueStorage) -> Option<ExternalValueKind> {
    match storage
    {
        ValueStorage::ExternalInput => Some(ExternalValueKind::Input),
        ValueStorage::ExternalConstant => Some(ExternalValueKind::Constant),
        ValueStorage::Buffer(_) => None,
    }
}

/// Lowers a canonical [`ExecutionPlan`] into a backend-neutral [`LoweredPlan`].
///
/// The lowerer performs no allocation, holds no device state and reads nothing
/// outside the plan and the binding table it is given.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KernelLowerer;

impl KernelLowerer {
    pub const fn new() -> Self {
        Self
    }

    /// Lowers `plan`, using `bindings` as the external calling order.
    ///
    /// Fails on the first inconsistency; a plan lowers completely or not at all.
    pub fn lower(
        self,
        plan: &ExecutionPlan,
        bindings: &ExternalBindings,
    ) -> Result<LoweredPlan, LoweringError> {
        let values = collect_values(plan)?;
        let binding_ids = validate_bindings(&values, bindings)?;

        let mut kernels = Vec::new();
        let mut instructions = Vec::new();

        for instruction in plan.instructions()
        {
            if matches!(
                instruction.operation,
                Operation::Input { .. } | Operation::Constant { .. }
            )
            {
                continue;
            }

            instructions.push(lower_instruction(
                instruction,
                &values,
                &binding_ids,
                &mut kernels,
            )?);
        }

        let mut outputs = Vec::with_capacity(plan.outputs().len());
        for &node in plan.outputs()
        {
            let info = values
                .get(&node)
                .ok_or(LoweringError::UnloweredOutput { node })?;

            outputs.push(LoweredOutput {
                node,
                source: source_of(node, info.storage, &binding_ids)?,
            });
        }

        Ok(LoweredPlan {
            kernels,
            instructions,
            bindings: bindings.entries().to_vec(),
            outputs,
        })
    }
}

/// Joins the retained instructions with their storage assignments.
fn collect_values(plan: &ExecutionPlan) -> Result<BTreeMap<NodeId, ValueInfo>, LoweringError> {
    let mut storages = BTreeMap::<NodeId, ValueStorage>::new();
    for allocation in plan.memory_plan().allocations()
    {
        storages.insert(allocation.node, allocation.storage);
    }

    let mut values = BTreeMap::<NodeId, ValueInfo>::new();
    for instruction in plan.instructions()
    {
        let storage =
            storages
                .get(&instruction.id)
                .copied()
                .ok_or(LoweringError::MissingBufferSlot {
                    node: instruction.id,
                })?;

        values.insert(
            instruction.id,
            ValueInfo {
                storage,
                tensor_type: instruction.output.clone(),
            },
        );
    }

    Ok(values)
}

/// Checks the binding table against the plan and indexes it by node.
fn validate_bindings(
    values: &BTreeMap<NodeId, ValueInfo>,
    bindings: &ExternalBindings,
) -> Result<BTreeMap<NodeId, LogicalBindingId>, LoweringError> {
    let mut binding_ids = BTreeMap::<NodeId, LogicalBindingId>::new();

    for (position, entry) in bindings.entries().iter().enumerate()
    {
        let info = values
            .get(&entry.node)
            .ok_or(LoweringError::UnknownNode { node: entry.node })?;

        let expected = external_kind(info.storage)
            .ok_or(LoweringError::ExtraneousExternalBinding { node: entry.node })?;

        if expected != entry.kind
        {
            return Err(LoweringError::ExternalBindingKindMismatch {
                node: entry.node,
                expected,
                actual: entry.kind,
            });
        }

        let raw = u32::try_from(position).map_err(|_| LoweringError::IdentifierOverflow {
            space: IdentifierSpace::ExternalBinding,
            node: entry.node,
        })?;

        if binding_ids
            .insert(entry.node, LogicalBindingId::new(raw))
            .is_some()
        {
            return Err(LoweringError::DuplicateExternalBinding { node: entry.node });
        }
    }

    for (&node, info) in values
    {
        if external_kind(info.storage).is_some() && !binding_ids.contains_key(&node)
        {
            return Err(LoweringError::MissingExternalBinding { node });
        }
    }

    Ok(binding_ids)
}

/// Logical value filling a reference to `node`.
fn source_of(
    node: NodeId,
    storage: ValueStorage,
    binding_ids: &BTreeMap<NodeId, LogicalBindingId>,
) -> Result<KernelArgumentSource, LoweringError> {
    match storage
    {
        ValueStorage::Buffer(slot) => Ok(KernelArgumentSource::Buffer(slot)),
        ValueStorage::ExternalInput => binding_ids
            .get(&node)
            .copied()
            .map(KernelArgumentSource::ExternalInput)
            .ok_or(LoweringError::MissingExternalBinding { node }),
        ValueStorage::ExternalConstant => binding_ids
            .get(&node)
            .copied()
            .map(KernelArgumentSource::ExternalConstant)
            .ok_or(LoweringError::MissingExternalBinding { node }),
    }
}

/// Lowers one computed instruction into one dispatch.
fn lower_instruction(
    instruction: &Instruction,
    values: &BTreeMap<NodeId, ValueInfo>,
    binding_ids: &BTreeMap<NodeId, LogicalBindingId>,
    kernels: &mut Vec<LogicalKernel>,
) -> Result<LoweredInstruction, LoweringError> {
    let node = instruction.id;

    let expected_arity = instruction.operation.expected_arity();
    if instruction.inputs.len() != expected_arity
    {
        return Err(LoweringError::ArityMismatch {
            node,
            expected: expected_arity,
            actual: instruction.inputs.len(),
        });
    }

    let mut operands = Vec::with_capacity(instruction.inputs.len());
    for &input in &instruction.inputs
    {
        let info = values
            .get(&input)
            .ok_or(LoweringError::UnknownNode { node: input })?;

        operands.push(info.tensor_type.clone());
    }

    let elements = checked_elements(node, &instruction.output.shape)?;
    let family = kernel_family(node, instruction, &operands, elements)?;

    let mut arguments = Vec::with_capacity(operands.len() + 1);
    for (position, (&input, tensor_type)) in instruction.inputs.iter().zip(&operands).enumerate()
    {
        let info = values
            .get(&input)
            .ok_or(LoweringError::UnknownNode { node: input })?;

        arguments.push(KernelArgument {
            index: argument_index(node, position)?,
            source: source_of(input, info.storage, binding_ids)?,
            access: KernelArgumentAccess::Read,
            tensor_type: tensor_type.clone(),
        });
    }

    let result_info = values
        .get(&node)
        .ok_or(LoweringError::MissingBufferSlot { node })?;

    arguments.push(KernelArgument {
        index: argument_index(node, arguments.len())?,
        source: source_of(node, result_info.storage, binding_ids)?,
        access: KernelArgumentAccess::Write,
        tensor_type: instruction.output.clone(),
    });

    for (position, argument) in arguments.iter().enumerate()
    {
        let expected = argument_index(node, position)?;
        if argument.index != expected
        {
            return Err(LoweringError::ArgumentIndexMismatch {
                node,
                expected,
                actual: argument.index,
            });
        }
    }

    let kernel = intern_kernel(kernels, node, family, operands, instruction.output.clone())?;

    Ok(LoweredInstruction {
        node,
        sources: vec![node],
        kernel,
        arguments,
        dispatch: LogicalDispatch {
            extent: instruction.output.shape.clone(),
            elements,
        },
    })
}

/// Selects the kernel family, validating the operation's semantics.
fn kernel_family(
    node: NodeId,
    instruction: &Instruction,
    operands: &[TensorType],
    elements: usize,
) -> Result<KernelFamily, LoweringError> {
    match &instruction.operation
    {
        Operation::Add => uniform_binary(node, BinaryKernel::Add, instruction, operands),
        Operation::Sub => uniform_binary(node, BinaryKernel::Sub, instruction, operands),
        Operation::Mul => uniform_binary(node, BinaryKernel::Mul, instruction, operands),
        Operation::Div => uniform_binary(node, BinaryKernel::Div, instruction, operands),

        Operation::Relu => uniform_unary(node, UnaryKernel::Relu, instruction, operands),
        Operation::Exp => uniform_unary(node, UnaryKernel::Exp, instruction, operands),
        Operation::Log => uniform_unary(node, UnaryKernel::Log, instruction, operands),
        Operation::Scale { factor } => uniform_unary(
            node,
            UnaryKernel::Scale { factor: *factor },
            instruction,
            operands,
        ),

        Operation::Reshape { shape } =>
        {
            let source = operands.first().ok_or(LoweringError::ArityMismatch {
                node,
                expected: 1,
                actual: operands.len(),
            })?;

            let source_elements = checked_elements(node, &source.shape)?;

            if source.dtype != instruction.output.dtype
                || instruction.output.shape != *shape
                || source_elements != elements
            {
                return Err(LoweringError::InvalidReshape { node });
            }

            Ok(KernelFamily::ShapeCopy)
        },

        Operation::Transpose { permutation } =>
        {
            let source = operands.first().ok_or(LoweringError::ArityMismatch {
                node,
                expected: 1,
                actual: operands.len(),
            })?;

            if !is_valid_permutation(permutation, source, &instruction.output)
            {
                return Err(LoweringError::InvalidPermutation { node });
            }

            Ok(KernelFamily::Permute {
                permutation: permutation.clone(),
            })
        },

        // `MatMul`, `Input`, `Constant` and any operation added later to the
        // non-exhaustive IR enum are rejected explicitly. No fallback.
        operation => Err(LoweringError::UnsupportedOperation {
            node,
            operation: operation.clone(),
        }),
    }
}

/// Element-wise binary kernel over two operands identical to the result.
fn uniform_binary(
    node: NodeId,
    kernel: BinaryKernel,
    instruction: &Instruction,
    operands: &[TensorType],
) -> Result<KernelFamily, LoweringError> {
    require_uniform_operands(node, instruction, operands)?;
    Ok(KernelFamily::ElementwiseBinary(kernel))
}

/// Element-wise unary kernel over one operand identical to the result.
fn uniform_unary(
    node: NodeId,
    kernel: UnaryKernel,
    instruction: &Instruction,
    operands: &[TensorType],
) -> Result<KernelFamily, LoweringError> {
    require_uniform_operands(node, instruction, operands)?;
    Ok(KernelFamily::ElementwiseUnary(kernel))
}

/// Every operand must carry exactly the result type.
fn require_uniform_operands(
    node: NodeId,
    instruction: &Instruction,
    operands: &[TensorType],
) -> Result<(), LoweringError> {
    for operand in operands
    {
        if *operand != instruction.output
        {
            return Err(LoweringError::OperandTypeMismatch {
                node,
                expected: instruction.output.clone(),
                actual: operand.clone(),
            });
        }
    }

    Ok(())
}

/// Validates `output.shape[i] == input.shape[permutation[i]]`.
fn is_valid_permutation(permutation: &[usize], input: &TensorType, output: &TensorType) -> bool {
    if input.dtype != output.dtype
    {
        return false;
    }

    let rank = input.shape.rank();
    if permutation.len() != rank || output.shape.rank() != rank
    {
        return false;
    }

    let mut axes = permutation.to_vec();
    axes.sort_unstable();
    if axes.iter().enumerate().any(|(axis, &value)| value != axis)
    {
        return false;
    }

    permutation
        .iter()
        .enumerate()
        .all(|(position, &axis)| output.shape.dims().get(position) == input.shape.dims().get(axis))
}

/// Element count of a shape, computed with the checked shape API.
fn checked_elements(node: NodeId, shape: &Shape) -> Result<usize, LoweringError> {
    shape
        .checked_num_elements()
        .map_err(|_| LoweringError::DispatchExtentOverflow { node })
}

/// Logical index of an argument position.
fn argument_index(node: NodeId, position: usize) -> Result<u32, LoweringError> {
    u32::try_from(position).map_err(|_| LoweringError::IdentifierOverflow {
        space: IdentifierSpace::KernelArgument,
        node,
    })
}

/// Returns the identifier of an identical existing kernel, or appends a new one.
///
/// Deduplication is a linear scan comparing full signatures, which is `O(n²)` in
/// the number of distinct kernels. That is accepted for this phase: no hash map
/// is used, because its iteration order would not be deterministic, and neither
/// [`TensorType`] nor [`Shape`] implements `Ord`.
fn intern_kernel(
    kernels: &mut Vec<LogicalKernel>,
    node: NodeId,
    family: KernelFamily,
    operands: Vec<TensorType>,
    result: TensorType,
) -> Result<LogicalKernelId, LoweringError> {
    let existing = kernels
        .iter()
        .find(|kernel| {
            kernel.family == family && kernel.operands == operands && kernel.result == result
        })
        .map(LogicalKernel::id);

    if let Some(id) = existing
    {
        return Ok(id);
    }

    let raw = u32::try_from(kernels.len()).map_err(|_| LoweringError::IdentifierOverflow {
        space: IdentifierSpace::LogicalKernel,
        node,
    })?;

    let id = LogicalKernelId::new(raw);
    kernels.push(LogicalKernel {
        id,
        family,
        operands,
        result,
    });

    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CanonicalCompiler;
    use scirust_tensor_ir::{ConstantId, DType, Graph};

    fn typed(dims: Vec<usize>) -> TensorType {
        TensorType::new(DType::F32, Shape::new(dims))
    }

    fn matrix() -> TensorType {
        typed(vec![2, 2])
    }

    fn compile(graph: &Graph) -> ExecutionPlan {
        CanonicalCompiler::new().compile(graph).unwrap()
    }

    fn lower(plan: &ExecutionPlan) -> Result<LoweredPlan, LoweringError> {
        let bindings = ExternalBindings::derive(plan);
        KernelLowerer::new().lower(plan, &bindings)
    }

    /// `input + constant`, then `relu`.
    fn sample_graph() -> Graph {
        let mut graph = Graph::new();

        let input = graph.add_input("x", matrix()).unwrap();
        let constant = graph.add_constant(ConstantId::new(3), matrix()).unwrap();
        let sum = graph
            .add_node(Operation::Add, vec![input, constant], matrix())
            .unwrap();
        let activated = graph
            .add_node(Operation::Relu, vec![sum], matrix())
            .unwrap();

        graph.set_outputs(vec![activated]).unwrap();
        graph
    }

    #[test]
    fn lowered_instructions_follow_canonical_order() {
        let plan = compile(&sample_graph());
        let lowered = lower(&plan).unwrap();

        assert_eq!(
            lowered
                .instructions()
                .iter()
                .map(|instruction| instruction.node)
                .collect::<Vec<_>>(),
            vec![NodeId::new(2), NodeId::new(3)]
        );

        assert!(matches!(
            lowered
                .kernel(lowered.instructions()[0].kernel)
                .unwrap()
                .family(),
            KernelFamily::ElementwiseBinary(BinaryKernel::Add)
        ));
        assert!(matches!(
            lowered
                .kernel(lowered.instructions()[1].kernel)
                .unwrap()
                .family(),
            KernelFamily::ElementwiseUnary(UnaryKernel::Relu)
        ));
    }

    #[test]
    fn arguments_are_reads_in_input_order_then_the_write() {
        let plan = compile(&sample_graph());
        let lowered = lower(&plan).unwrap();
        let add = lowered.instruction_for(NodeId::new(2)).unwrap();

        assert_eq!(
            add.arguments
                .iter()
                .map(|argument| (argument.index, argument.access))
                .collect::<Vec<_>>(),
            vec![
                (0, KernelArgumentAccess::Read),
                (1, KernelArgumentAccess::Read),
                (2, KernelArgumentAccess::Write),
            ]
        );

        // Read order mirrors `Instruction::inputs` exactly.
        assert_eq!(
            add.arguments[0].source,
            KernelArgumentSource::ExternalInput(LogicalBindingId::new(0))
        );
        assert_eq!(
            add.arguments[1].source,
            KernelArgumentSource::ExternalConstant(LogicalBindingId::new(1))
        );
    }

    #[test]
    fn external_inputs_and_constants_are_distinguished() {
        let plan = compile(&sample_graph());
        let lowered = lower(&plan).unwrap();

        assert_eq!(
            lowered.bindings(),
            &[
                ExternalBinding {
                    node: NodeId::new(0),
                    kind: ExternalValueKind::Input,
                },
                ExternalBinding {
                    node: NodeId::new(1),
                    kind: ExternalValueKind::Constant,
                },
            ]
        );

        assert_eq!(
            lowered.binding(LogicalBindingId::new(1)).unwrap().kind,
            ExternalValueKind::Constant
        );
    }

    #[test]
    fn logical_buffer_slots_are_preserved() {
        let plan = compile(&sample_graph());
        let lowered = lower(&plan).unwrap();

        for instruction in lowered.instructions()
        {
            let planned = plan.memory_plan().storage_for(instruction.node).unwrap();
            let write = instruction.arguments.last().unwrap();

            assert_eq!(write.access, KernelArgumentAccess::Write);
            assert_eq!(
                write.source,
                match planned
                {
                    ValueStorage::Buffer(slot) => KernelArgumentSource::Buffer(slot),
                    other => panic!("computed value should own a buffer, got {other:?}"),
                }
            );
        }
    }

    #[test]
    fn producer_node_identifiers_are_preserved() {
        let plan = compile(&sample_graph());
        let lowered = lower(&plan).unwrap();

        for instruction in lowered.instructions()
        {
            assert_eq!(instruction.sources, vec![instruction.node]);
            assert!(
                plan.instructions()
                    .iter()
                    .any(|canonical| canonical.id == instruction.node)
            );
        }
    }

    #[test]
    fn missing_external_binding_is_rejected() {
        let plan = compile(&sample_graph());
        let bindings = ExternalBindings::from_entries(vec![ExternalBinding {
            node: NodeId::new(0),
            kind: ExternalValueKind::Input,
        }]);

        assert_eq!(
            KernelLowerer::new().lower(&plan, &bindings),
            Err(LoweringError::MissingExternalBinding {
                node: NodeId::new(1)
            })
        );
    }

    #[test]
    fn duplicate_external_binding_is_rejected() {
        let plan = compile(&sample_graph());
        let bindings = ExternalBindings::from_entries(vec![
            ExternalBinding {
                node: NodeId::new(0),
                kind: ExternalValueKind::Input,
            },
            ExternalBinding {
                node: NodeId::new(0),
                kind: ExternalValueKind::Input,
            },
        ]);

        assert_eq!(
            KernelLowerer::new().lower(&plan, &bindings),
            Err(LoweringError::DuplicateExternalBinding {
                node: NodeId::new(0)
            })
        );
    }

    #[test]
    fn extraneous_external_binding_is_rejected() {
        let plan = compile(&sample_graph());
        let mut entries = ExternalBindings::derive(&plan).entries().to_vec();
        entries.push(ExternalBinding {
            node: NodeId::new(2),
            kind: ExternalValueKind::Input,
        });

        assert_eq!(
            KernelLowerer::new().lower(&plan, &ExternalBindings::from_entries(entries)),
            Err(LoweringError::ExtraneousExternalBinding {
                node: NodeId::new(2)
            })
        );
    }

    #[test]
    fn external_binding_kind_mismatch_is_rejected() {
        let plan = compile(&sample_graph());
        let bindings = ExternalBindings::from_entries(vec![
            ExternalBinding {
                node: NodeId::new(0),
                kind: ExternalValueKind::Constant,
            },
            ExternalBinding {
                node: NodeId::new(1),
                kind: ExternalValueKind::Constant,
            },
        ]);

        assert_eq!(
            KernelLowerer::new().lower(&plan, &bindings),
            Err(LoweringError::ExternalBindingKindMismatch {
                node: NodeId::new(0),
                expected: ExternalValueKind::Input,
                actual: ExternalValueKind::Constant,
            })
        );
    }

    #[test]
    fn unknown_node_in_the_binding_table_is_rejected() {
        let plan = compile(&sample_graph());
        let bindings = ExternalBindings::from_entries(vec![ExternalBinding {
            node: NodeId::new(99),
            kind: ExternalValueKind::Input,
        }]);

        assert_eq!(
            KernelLowerer::new().lower(&plan, &bindings),
            Err(LoweringError::UnknownNode {
                node: NodeId::new(99)
            })
        );
    }

    #[test]
    fn matmul_is_rejected_as_unsupported() {
        let mut graph = Graph::new();
        let lhs = graph.add_input("lhs", matrix()).unwrap();
        let rhs = graph.add_input("rhs", matrix()).unwrap();
        let product = graph
            .add_node(Operation::MatMul, vec![lhs, rhs], matrix())
            .unwrap();
        graph.set_outputs(vec![product]).unwrap();

        let plan = compile(&graph);

        assert_eq!(
            lower(&plan),
            Err(LoweringError::UnsupportedOperation {
                node: NodeId::new(2),
                operation: Operation::MatMul,
            })
        );
    }

    #[test]
    fn an_unsupported_operation_has_no_silent_fallback() {
        // The plan mixes supported operations with one unsupported operation.
        // Lowering must fail entirely rather than skip, no-op or partially emit.
        let mut graph = Graph::new();
        let lhs = graph.add_input("lhs", matrix()).unwrap();
        let rhs = graph.add_input("rhs", matrix()).unwrap();
        let sum = graph
            .add_node(Operation::Add, vec![lhs, rhs], matrix())
            .unwrap();
        let product = graph
            .add_node(Operation::MatMul, vec![sum, rhs], matrix())
            .unwrap();
        let activated = graph
            .add_node(Operation::Relu, vec![product], matrix())
            .unwrap();
        graph.set_outputs(vec![activated]).unwrap();

        let plan = compile(&graph);

        assert_eq!(plan.instructions().len(), 5);
        assert!(matches!(
            lower(&plan),
            Err(LoweringError::UnsupportedOperation { .. })
        ));
    }

    #[test]
    fn repeated_lowering_of_the_same_plan_is_identical() {
        let plan = compile(&sample_graph());

        let first = lower(&plan).unwrap();
        let second = lower(&plan).unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn eliminated_dead_nodes_leave_non_contiguous_node_identifiers() {
        let mut graph = Graph::new();

        let input = graph.add_input("x", matrix()).unwrap();
        let dead_constant = graph.add_constant(ConstantId::new(9), matrix()).unwrap();
        let _dead = graph
            .add_node(Operation::Exp, vec![dead_constant], matrix())
            .unwrap();
        let live = graph
            .add_node(Operation::Relu, vec![input], matrix())
            .unwrap();

        graph.set_outputs(vec![live]).unwrap();

        let plan = compile(&graph);
        assert_eq!(plan.stats().eliminated_nodes, 2);

        let lowered = lower(&plan).unwrap();

        // Node 3 survives while nodes 1 and 2 are gone: identifiers are not
        // positions, and nothing may index by `NodeId::get`.
        assert_eq!(lowered.instructions().len(), 1);
        assert_eq!(lowered.instructions()[0].node, NodeId::new(3));
        assert_eq!(
            lowered.bindings(),
            &[ExternalBinding {
                node: NodeId::new(0),
                kind: ExternalValueKind::Input,
            }]
        );
        assert_eq!(
            lowered.instructions()[0].arguments[0].source,
            KernelArgumentSource::ExternalInput(LogicalBindingId::new(0))
        );
    }

    #[test]
    fn output_order_is_preserved() {
        let mut graph = Graph::new();

        let input = graph.add_input("x", matrix()).unwrap();
        let first = graph
            .add_node(Operation::Relu, vec![input], matrix())
            .unwrap();
        let second = graph
            .add_node(Operation::Exp, vec![input], matrix())
            .unwrap();

        // Declared in reverse construction order, and an external input is an
        // output too.
        graph.set_outputs(vec![second, first, input]).unwrap();

        let plan = compile(&graph);
        let lowered = lower(&plan).unwrap();

        assert_eq!(
            lowered
                .outputs()
                .iter()
                .map(|output| output.node)
                .collect::<Vec<_>>(),
            vec![second, first, input]
        );

        // An output that is an external input carries no kernel.
        assert_eq!(
            lowered.outputs()[2].source,
            KernelArgumentSource::ExternalInput(LogicalBindingId::new(0))
        );
        assert!(lowered.instruction_for(input).is_none());
    }

    #[test]
    fn identical_kernels_are_deduplicated_deterministically() {
        let mut graph = Graph::new();

        let input = graph.add_input("x", matrix()).unwrap();
        let first = graph
            .add_node(Operation::Relu, vec![input], matrix())
            .unwrap();
        let second = graph
            .add_node(Operation::Relu, vec![first], matrix())
            .unwrap();

        graph.set_outputs(vec![second]).unwrap();

        let plan = compile(&graph);
        let lowered = lower(&plan).unwrap();

        assert_eq!(lowered.kernels().len(), 1);
        assert_eq!(lowered.kernels()[0].id(), LogicalKernelId::new(0));
        assert_eq!(
            lowered
                .instructions()
                .iter()
                .map(|instruction| instruction.kernel)
                .collect::<Vec<_>>(),
            vec![LogicalKernelId::new(0), LogicalKernelId::new(0)]
        );
    }

    #[test]
    fn kernels_differing_only_by_type_are_not_deduplicated() {
        let mut graph = Graph::new();

        let small = graph.add_input("small", typed(vec![2])).unwrap();
        let large = graph.add_input("large", typed(vec![4])).unwrap();
        let first = graph
            .add_node(Operation::Relu, vec![small], typed(vec![2]))
            .unwrap();
        let second = graph
            .add_node(Operation::Relu, vec![large], typed(vec![4]))
            .unwrap();
        let third = graph
            .add_node(Operation::Relu, vec![first], typed(vec![2]))
            .unwrap();

        graph.set_outputs(vec![second, third]).unwrap();

        let plan = compile(&graph);
        let lowered = lower(&plan).unwrap();

        assert_eq!(lowered.kernels().len(), 2);
        assert_eq!(
            lowered
                .instructions()
                .iter()
                .map(|instruction| instruction.kernel)
                .collect::<Vec<_>>(),
            vec![
                LogicalKernelId::new(0),
                LogicalKernelId::new(1),
                LogicalKernelId::new(0),
            ]
        );
    }

    #[test]
    fn scale_keeps_its_factor_inside_the_kernel_signature() {
        let mut graph = Graph::new();

        let input = graph.add_input("x", matrix()).unwrap();
        let scaled = graph
            .add_node(
                Operation::Scale {
                    factor: Scalar::f32(0.5),
                },
                vec![input],
                matrix(),
            )
            .unwrap();

        graph.set_outputs(vec![scaled]).unwrap();

        let plan = compile(&graph);
        let lowered = lower(&plan).unwrap();

        assert_eq!(
            lowered.kernels()[0].family(),
            &KernelFamily::ElementwiseUnary(UnaryKernel::Scale {
                factor: Scalar::f32(0.5),
            })
        );

        // The factor is not an external binding.
        assert_eq!(lowered.bindings().len(), 1);
        assert_eq!(lowered.instructions()[0].arguments.len(), 2);
    }

    #[test]
    fn valid_reshape_lowers_to_a_shape_copy() {
        let mut graph = Graph::new();

        let input = graph.add_input("x", typed(vec![2, 3])).unwrap();
        let reshaped = graph
            .add_node(
                Operation::Reshape {
                    shape: Shape::new(vec![6]),
                },
                vec![input],
                typed(vec![6]),
            )
            .unwrap();

        graph.set_outputs(vec![reshaped]).unwrap();

        let plan = compile(&graph);
        let lowered = lower(&plan).unwrap();
        let instruction = &lowered.instructions()[0];

        assert_eq!(
            lowered.kernel(instruction.kernel).unwrap().family(),
            &KernelFamily::ShapeCopy
        );
        assert_eq!(instruction.dispatch.extent, Shape::new(vec![6]));
        assert_eq!(instruction.dispatch.elements, 6);
        assert_eq!(instruction.arguments.len(), 2);

        // The result is written into the distinct slot the memory plan reserved.
        assert_eq!(
            instruction.arguments[1].source,
            KernelArgumentSource::Buffer(BufferSlot::new(0))
        );
    }

    #[test]
    fn inconsistent_reshape_is_rejected() {
        let mut graph = Graph::new();

        let input = graph.add_input("x", typed(vec![2, 3])).unwrap();
        // Six elements cannot be reshaped into three.
        let reshaped = graph
            .add_node(
                Operation::Reshape {
                    shape: Shape::new(vec![3]),
                },
                vec![input],
                typed(vec![3]),
            )
            .unwrap();

        graph.set_outputs(vec![reshaped]).unwrap();

        let plan = compile(&graph);

        assert_eq!(
            lower(&plan),
            Err(LoweringError::InvalidReshape {
                node: NodeId::new(1)
            })
        );
    }

    #[test]
    fn reshape_disagreeing_with_its_declared_output_is_rejected() {
        let mut graph = Graph::new();

        let input = graph.add_input("x", typed(vec![2, 3])).unwrap();
        // The element count matches, but the declared output shape does not
        // match the shape carried by the operation.
        let reshaped = graph
            .add_node(
                Operation::Reshape {
                    shape: Shape::new(vec![6]),
                },
                vec![input],
                typed(vec![3, 2]),
            )
            .unwrap();

        graph.set_outputs(vec![reshaped]).unwrap();

        let plan = compile(&graph);

        assert_eq!(
            lower(&plan),
            Err(LoweringError::InvalidReshape {
                node: NodeId::new(1)
            })
        );
    }

    #[test]
    fn valid_transpose_lowers_to_a_permutation() {
        let mut graph = Graph::new();

        let input = graph.add_input("x", typed(vec![2, 3])).unwrap();
        let transposed = graph
            .add_node(
                Operation::Transpose {
                    permutation: vec![1, 0],
                },
                vec![input],
                typed(vec![3, 2]),
            )
            .unwrap();

        graph.set_outputs(vec![transposed]).unwrap();

        let plan = compile(&graph);
        let lowered = lower(&plan).unwrap();
        let instruction = &lowered.instructions()[0];

        assert_eq!(
            lowered.kernel(instruction.kernel).unwrap().family(),
            &KernelFamily::Permute {
                permutation: vec![1, 0],
            }
        );
        assert_eq!(instruction.dispatch.extent, Shape::new(vec![3, 2]));
        assert_eq!(instruction.dispatch.elements, 6);
    }

    #[test]
    fn transpose_with_a_repeated_axis_is_rejected() {
        let mut graph = Graph::new();

        let input = graph.add_input("x", typed(vec![2, 2])).unwrap();
        let transposed = graph
            .add_node(
                Operation::Transpose {
                    permutation: vec![0, 0],
                },
                vec![input],
                typed(vec![2, 2]),
            )
            .unwrap();

        graph.set_outputs(vec![transposed]).unwrap();

        let plan = compile(&graph);

        assert_eq!(
            lower(&plan),
            Err(LoweringError::InvalidPermutation {
                node: NodeId::new(1)
            })
        );
    }

    #[test]
    fn transpose_disagreeing_with_the_permuted_shape_is_rejected() {
        let mut graph = Graph::new();

        let input = graph.add_input("x", typed(vec![2, 3])).unwrap();
        // Under `output.shape[i] == input.shape[permutation[i]]` the result must
        // be `[3, 2]`, not `[2, 3]`.
        let transposed = graph
            .add_node(
                Operation::Transpose {
                    permutation: vec![1, 0],
                },
                vec![input],
                typed(vec![2, 3]),
            )
            .unwrap();

        graph.set_outputs(vec![transposed]).unwrap();

        let plan = compile(&graph);

        assert_eq!(
            lower(&plan),
            Err(LoweringError::InvalidPermutation {
                node: NodeId::new(1)
            })
        );
    }

    #[test]
    fn operand_type_mismatch_is_rejected() {
        let mut graph = Graph::new();

        // `Graph::validate` checks arity and reference direction only, so this
        // inconsistent graph reaches the lowerer.
        let lhs = graph.add_input("lhs", typed(vec![2, 2])).unwrap();
        let rhs = graph.add_input("rhs", typed(vec![3, 3])).unwrap();
        let sum = graph
            .add_node(Operation::Add, vec![lhs, rhs], typed(vec![2, 2]))
            .unwrap();

        graph.set_outputs(vec![sum]).unwrap();
        assert_eq!(graph.validate(), Ok(()));

        let plan = compile(&graph);

        assert_eq!(
            lower(&plan),
            Err(LoweringError::OperandTypeMismatch {
                node: NodeId::new(2),
                expected: typed(vec![2, 2]),
                actual: typed(vec![3, 3]),
            })
        );
    }

    #[test]
    fn repeated_operand_yields_two_read_arguments_on_one_slot() {
        let mut graph = Graph::new();

        let input = graph.add_input("x", matrix()).unwrap();
        let activated = graph
            .add_node(Operation::Relu, vec![input], matrix())
            .unwrap();
        let doubled = graph
            .add_node(Operation::Add, vec![activated, activated], matrix())
            .unwrap();

        graph.set_outputs(vec![doubled]).unwrap();

        let plan = compile(&graph);
        let lowered = lower(&plan).unwrap();
        let add = lowered.instruction_for(doubled).unwrap();

        assert_eq!(add.arguments[0].source, add.arguments[1].source);
        assert_ne!(add.arguments[0].index, add.arguments[1].index);
        assert_eq!(add.arguments[0].access, KernelArgumentAccess::Read);
        assert_eq!(add.arguments[1].access, KernelArgumentAccess::Read);

        // No supported family ever needs read-write access.
        assert!(
            add.arguments
                .iter()
                .all(|argument| argument.access != KernelArgumentAccess::Write
                    || argument.source != add.arguments[0].source)
        );
    }

    #[test]
    fn caller_chosen_binding_order_defines_the_binding_identifiers() {
        let plan = compile(&sample_graph());
        let reversed = ExternalBindings::from_entries(vec![
            ExternalBinding {
                node: NodeId::new(1),
                kind: ExternalValueKind::Constant,
            },
            ExternalBinding {
                node: NodeId::new(0),
                kind: ExternalValueKind::Input,
            },
        ]);

        let lowered = KernelLowerer::new().lower(&plan, &reversed).unwrap();
        let add = lowered.instruction_for(NodeId::new(2)).unwrap();

        assert_eq!(
            add.arguments[0].source,
            KernelArgumentSource::ExternalInput(LogicalBindingId::new(1))
        );
        assert_eq!(
            add.arguments[1].source,
            KernelArgumentSource::ExternalConstant(LogicalBindingId::new(0))
        );
    }

    #[test]
    fn errors_have_stable_messages() {
        assert_eq!(
            LoweringError::MissingExternalBinding {
                node: NodeId::new(4)
            }
            .to_string(),
            "external value of node 4 is not bound"
        );

        assert_eq!(
            LoweringError::InvalidPermutation {
                node: NodeId::new(7)
            }
            .to_string(),
            "node 7 declares an inconsistent transpose permutation"
        );
    }
}
