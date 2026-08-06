//! Executes a whole `LoweredPlan` on a backend that speaks the Reference
//! kernel format.
//!
//! # Two phases
//!
//! [`ReferencePlanRuntime::prepare`] validates a plan, generates its Reference
//! artefacts, compiles each logical kernel once, and returns an immutable
//! [`PreparedReferencePlan`]. [`ReferencePlanRuntime::execute`] then runs that
//! prepared plan over caller-supplied values, as often as wanted, with
//! different values each time. Nothing is generated, compiled or re-validated
//! per execution.
//!
//! # Backend requirements
//!
//! The runtime is generic over [`ComputeBackend`], but that does **not** mean
//! every backend can run a Reference plan. `DeviceCapabilities` exposes neither
//! an alignment nor a memory space, so this runtime cannot negotiate them; it
//! allocates with `alignment = 1` and [`MemorySpace::Host`], the placement the
//! CPU Reference adapter requires. A backend handed to this runtime must
//! therefore accept:
//!
//! * [`scirust_compute::KernelFormat::Reference`] modules;
//! * [`MemorySpace::Host`] with an alignment of `1`;
//! * [`DType::F32`];
//! * the canonical `[1, 1, 1]` grid and block with no shared memory.
//!
//! What *can* be checked against the contract is checked at preparation: `F32`
//! support and the workgroup limit come from `capabilities()`, and buffer sizes
//! are validated against `max_buffer_bytes`. The memory space and alignment
//! cannot be, so `allocate` reports them as a typed error at execution instead.
//! No advertised backend limit is ever bypassed.
//!
//! # Determinism
//!
//! This module contains no `HashMap`. Identifiers are dense — a
//! [`LogicalKernelId`] is a kernel's position, a [`LogicalBindingId`] is a
//! binding's position — so every table is a `Vec` indexed by identifier, and
//! every traversal follows a canonical order:
//!
//! * kernels are generated and compiled in `LogicalKernelId` order;
//! * buffer slots are allocated in [`BufferSlot`] order;
//! * external values are imported in [`LogicalBindingId`] order;
//! * dispatches run in the exact order of `LoweredPlan::instructions`;
//! * outputs are read back in the exact order of `LoweredPlan::outputs`.
//!
//! The order in which a caller supplies external values has no effect: they are
//! resolved by identifier. There is no clock, no random identifier and no
//! concurrency anywhere in this runtime. Numeric behaviour is entirely the
//! backend's; this runtime only moves bytes and issues dispatches.
//!
//! # Byte ABI
//!
//! Values cross the backend boundary as little-endian four-byte `f32` groups,
//! matching the Reference wire format and the CPU adapter's buffer ABI. No
//! slice is ever cast, and this crate contains no `unsafe`.

use scirust_compute::{
    BufferAccess, BufferBinding, ComputeBackend, DType, LaunchConfig, MemorySpace,
};
use scirust_tensor_compile::{
    BufferSlot, ExternalValueKind, KernelArgumentAccess, KernelArgumentSource, LogicalBindingId,
    LogicalKernelId, LoweredPlan,
};
use scirust_tensor_ir::{NodeId, TensorType};
use scirust_tensor_reference::{ReferenceKernelGenerator, ReferenceOpcode};

use crate::error::{PlanExecutionError, PlanPreparationError};

/// Storage width of one `f32` in a Reference buffer.
const F32_BYTES: usize = 4;

/// Allocation placement required by the Reference CPU backend.
const REFERENCE_ALIGNMENT: usize = 1;
const REFERENCE_MEMORY_SPACE: MemorySpace = MemorySpace::Host;

/// The launch geometry the Reference backend accepts.
const CANONICAL_LAUNCH_CONFIG: LaunchConfig = LaunchConfig {
    grid: [1, 1, 1],
    block: [1, 1, 1],
    shared_memory_bytes: 0,
};

// ---------------------------------------------------------------------------
// Public metadata
// ---------------------------------------------------------------------------

/// One external value a prepared plan needs before it can run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalValueSpec {
    pub binding: LogicalBindingId,
    pub node: NodeId,
    pub kind: ExternalValueKind,
    pub tensor_type: TensorType,
}

/// One internal buffer slot a prepared plan materialises per execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferSlotSpecification {
    pub slot: BufferSlot,
    pub tensor_type: TensorType,
}

/// One value a prepared plan produces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanOutputSpec {
    pub node: NodeId,
    pub tensor_type: TensorType,
}

/// One produced value, owned by the caller.
///
/// Holds no borrow of any backend buffer: the bytes were read out and decoded
/// before this value existed.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanOutputValue {
    pub node: NodeId,
    pub tensor_type: TensorType,
    pub values: Vec<f32>,
}

/// The values a plan produced, in the plan's declared output order.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanOutputs {
    values: Vec<PlanOutputValue>,
}

impl PlanOutputs {
    pub fn values(&self) -> &[PlanOutputValue] {
        &self.values
    }

    pub fn into_values(self) -> Vec<PlanOutputValue> {
        self.values
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// The external inputs *and* external constants a plan execution needs.
///
/// Both are supplied here because a `LoweredPlan` carries neither input names
/// nor `ConstantId`s — only a `NodeId` and an [`ExternalValueKind`] per binding.
/// Values are therefore addressed by [`LogicalBindingId`], which
/// [`PreparedReferencePlan::external_values`] enumerates.
///
/// Insertion order has no functional effect: everything is resolved by
/// identifier. A duplicate binding is accepted while building and rejected by
/// [`ReferencePlanRuntime::execute`], before any allocation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PlanExternalValues<'a> {
    entries: Vec<(LogicalBindingId, &'a [f32])>,
}

impl<'a> PlanExternalValues<'a> {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Supplies one external value. Chainable.
    pub fn bind(&mut self, binding: LogicalBindingId, values: &'a [f32]) -> &mut Self {
        self.entries.push((binding, values));
        self
    }

    pub fn entries(&self) -> &[(LogicalBindingId, &'a [f32])] {
        &self.entries
    }
}

// ---------------------------------------------------------------------------
// Prepared plan
// ---------------------------------------------------------------------------

/// Where one value physically lives during an execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PhysicalTarget {
    Slot(usize),
    External(usize),
}

/// Access of a resolved argument.
///
/// `KernelArgumentAccess` is `#[non_exhaustive]`, so it is narrowed once, at
/// preparation, into this closed two-valued form. The execution path then
/// matches exhaustively and can never encounter an access kind it would have to
/// guess about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreparedAccess {
    Read,
    Write,
}

/// One resolved kernel argument.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PreparedArgument {
    binding_slot: u32,
    target: PhysicalTarget,
    length_bytes: usize,
    access: PreparedAccess,
}

/// One resolved dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PreparedDispatch {
    node: NodeId,
    kernel: LogicalKernelId,
    arguments: Vec<PreparedArgument>,
}

/// One resolved output.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PreparedOutput {
    spec: PlanOutputSpec,
    target: PhysicalTarget,
    length_bytes: usize,
}

/// A validated plan with every kernel already compiled.
///
/// Immutable and self-contained: it borrows nothing from the `LoweredPlan` it
/// came from and holds no execution state — no buffer, no stream, no event, no
/// previously supplied value and no previous output. One prepared plan can back
/// any number of independent executions.
#[derive(Debug)]
pub struct PreparedReferencePlan<K> {
    kernels: Vec<K>,
    dispatches: Vec<PreparedDispatch>,
    slots: Vec<BufferSlotSpecification>,
    externals: Vec<ExternalValueSpec>,
    outputs: Vec<PreparedOutput>,
    slot_sizes: Vec<usize>,
    external_sizes: Vec<usize>,
    launch_config: LaunchConfig,
}

impl<K> PreparedReferencePlan<K> {
    /// Number of distinct compiled kernels — one per [`LogicalKernelId`], not
    /// one per dispatch.
    pub fn kernel_count(&self) -> usize {
        self.kernels.len()
    }

    pub fn dispatch_count(&self) -> usize {
        self.dispatches.len()
    }

    /// External values this plan needs, in [`LogicalBindingId`] order.
    pub fn external_values(&self) -> &[ExternalValueSpec] {
        &self.externals
    }

    /// Internal buffer slots, in [`BufferSlot`] order.
    pub fn buffer_slots(&self) -> &[BufferSlotSpecification] {
        &self.slots
    }

    /// Outputs this plan produces, in the plan's declared order.
    pub fn outputs(&self) -> Vec<PlanOutputSpec> {
        self.outputs
            .iter()
            .map(|output| output.spec.clone())
            .collect()
    }

    /// The launch geometry every dispatch uses.
    pub const fn launch_config(&self) -> LaunchConfig {
        self.launch_config
    }
}

// ---------------------------------------------------------------------------
// Runtime
// ---------------------------------------------------------------------------

/// Runs `LoweredPlan`s on a Reference-capable [`ComputeBackend`].
///
/// See the module documentation for the requirements a backend must satisfy and
/// for the determinism this runtime guarantees.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReferencePlanRuntime<B> {
    backend: B,
}

impl<B: ComputeBackend> ReferencePlanRuntime<B> {
    pub const fn new(backend: B) -> Self {
        Self { backend }
    }

    pub const fn backend(&self) -> &B {
        &self.backend
    }

    /// Validates `plan`, generates its Reference artefacts once, and compiles
    /// each logical kernel exactly once.
    ///
    /// Equivalent to [`Self::prepare_with_external_types`] with an empty hint
    /// list: a plan whose external value has no tensor type anywhere is rejected
    /// with [`PlanPreparationError::UndeterminedValueType`], exactly as before
    /// hints existed.
    pub fn prepare(
        &self,
        plan: &LoweredPlan,
    ) -> Result<PreparedReferencePlan<B::Kernel>, PlanPreparationError> {
        self.prepare_with_external_types(plan, &[])
    }

    /// Prepares `plan`, completing missing external metadata from
    /// `external_types`.
    ///
    /// A `LoweredPlan` carries a tensor type only on kernel arguments, so an
    /// external value that is *only* a plan output — never consumed by any
    /// dispatch — has no type anywhere in the plan. A caller that holds the
    /// canonical `Graph` does know that type, and supplies it here as a
    /// `(NodeId, TensorType)` pair.
    ///
    /// Hints complete metadata; they never invent or override anything:
    ///
    /// * the node must already be an external binding of `plan`
    ///   ([`PlanPreparationError::UnknownExternalTypeNode`] otherwise);
    /// * a node may be supplied at most once
    ///   ([`PlanPreparationError::DuplicateExternalType`] otherwise);
    /// * the type must be `F32` ([`PlanPreparationError::UnsupportedDType`]
    ///   otherwise);
    /// * where the plan already states a type through a kernel argument, both
    ///   must be *exactly* equal
    ///   ([`PlanPreparationError::ExternalTypeContradiction`] otherwise).
    ///
    /// No hint adds a binding, changes a binding's kind or `NodeId`, or reorders
    /// the binding table: the table is `plan`'s, untouched.
    pub fn prepare_with_external_types(
        &self,
        plan: &LoweredPlan,
        external_types: &[(NodeId, TensorType)],
    ) -> Result<PreparedReferencePlan<B::Kernel>, PlanPreparationError> {
        let capabilities = self.backend.capabilities();
        if !capabilities.supports_dtype(DType::F32)
        {
            return Err(PlanPreparationError::BackendDTypeUnsupported);
        }
        if capabilities
            .max_workgroup_size
            .iter()
            .any(|dimension| *dimension < 1)
        {
            return Err(PlanPreparationError::BackendWorkgroupTooSmall {
                max_workgroup_size: capabilities.max_workgroup_size,
            });
        }
        let max_buffer_bytes = capabilities.max_buffer_bytes;

        let types = collect_value_types(plan, external_types)?;
        let dispatches = resolve_dispatches(plan, &types)?;
        let (slots, slot_sizes) = build_slot_specifications(&types, max_buffer_bytes)?;
        let (externals, external_sizes) =
            build_external_specifications(plan, &types, max_buffer_bytes)?;
        let outputs = resolve_outputs(plan, &types)?;

        let kernels = compile_kernels(&self.backend, plan)?;

        Ok(PreparedReferencePlan {
            kernels,
            dispatches,
            slots,
            externals,
            outputs,
            slot_sizes,
            external_sizes,
            launch_config: CANONICAL_LAUNCH_CONFIG,
        })
    }

    /// Runs a prepared plan over `values`.
    ///
    /// Every allocation, buffer, stream and event belongs to this call alone, so
    /// a failure leaves the prepared plan untouched and reusable. Outputs are
    /// read back and assembled only after every dispatch, every wait and the
    /// final synchronisation have succeeded, so a failed run returns no partial
    /// [`PlanOutputs`] and never mutates the caller's slices.
    pub fn execute(
        &self,
        prepared: &PreparedReferencePlan<B::Kernel>,
        values: &PlanExternalValues<'_>,
    ) -> Result<PlanOutputs, PlanExecutionError> {
        // 1. Validate every supplied value before allocating anything.
        let supplied = resolve_external_values(prepared, values)?;

        // 2. Buffer slots, in slot order.
        let mut slot_buffers = Vec::with_capacity(prepared.slot_sizes.len());
        for &bytes in &prepared.slot_sizes
        {
            slot_buffers.push(self.allocate(bytes)?);
        }

        // 3. External buffers, in binding order.
        let mut external_buffers = Vec::with_capacity(prepared.external_sizes.len());
        for &bytes in &prepared.external_sizes
        {
            external_buffers.push(self.allocate(bytes)?);
        }

        // 4. Import external values, in binding order.
        for (index, value) in supplied.iter().enumerate()
        {
            let buffer = external_buffers
                .get(index)
                .ok_or(PlanExecutionError::MissingPhysicalBuffer { dispatch_index: 0 })?;
            let binding = LogicalBindingId::new(index_as_u32(index));

            self.backend
                .write(buffer, 0, &encode_le(value))
                .map_err(|source| PlanExecutionError::BufferWrite { binding, source })?;
        }

        // 5. One stream for this execution.
        let stream = self
            .backend
            .create_stream()
            .map_err(|source| PlanExecutionError::StreamCreation { source })?;

        // 6. Dispatches, strictly in plan order.
        for (dispatch_index, dispatch) in prepared.dispatches.iter().enumerate()
        {
            let kernel = prepared.kernels.get(dispatch.kernel.get() as usize).ok_or(
                PlanExecutionError::MissingCompiledKernel {
                    dispatch_index,
                    kernel: dispatch.kernel,
                },
            )?;

            let mut bindings = Vec::with_capacity(dispatch.arguments.len());
            for argument in &dispatch.arguments
            {
                let buffer = match argument.target
                {
                    PhysicalTarget::Slot(index) => slot_buffers.get(index),
                    PhysicalTarget::External(index) => external_buffers.get(index),
                }
                .ok_or(PlanExecutionError::MissingPhysicalBuffer { dispatch_index })?;

                bindings.push(BufferBinding {
                    slot: argument.binding_slot,
                    buffer,
                    offset_bytes: 0,
                    length_bytes: argument.length_bytes,
                    access: match argument.access
                    {
                        PreparedAccess::Read => BufferAccess::ReadOnly,
                        PreparedAccess::Write => BufferAccess::WriteOnly,
                    },
                });
            }

            let event = self
                .backend
                .launch(kernel, &stream, prepared.launch_config, &bindings)
                .map_err(|source| PlanExecutionError::KernelLaunch {
                    dispatch_index,
                    node: dispatch.node,
                    kernel: dispatch.kernel,
                    source,
                })?;

            self.backend
                .wait(&event)
                .map_err(|source| PlanExecutionError::EventWait {
                    dispatch_index,
                    node: dispatch.node,
                    kernel: dispatch.kernel,
                    source,
                })?;
        }

        // 7. Synchronise before reading anything back.
        self.backend
            .synchronize(&stream)
            .map_err(|source| PlanExecutionError::Synchronization { source })?;

        // 8. Read outputs, in plan order, only now that everything succeeded.
        let mut outputs = Vec::with_capacity(prepared.outputs.len());
        for (output_index, output) in prepared.outputs.iter().enumerate()
        {
            let buffer = match output.target
            {
                PhysicalTarget::Slot(index) => slot_buffers.get(index),
                PhysicalTarget::External(index) => external_buffers.get(index),
            }
            .ok_or(PlanExecutionError::MissingPhysicalBuffer {
                dispatch_index: output_index,
            })?;

            let mut bytes = vec![0u8; output.length_bytes];
            self.backend.read(buffer, 0, &mut bytes).map_err(|source| {
                PlanExecutionError::BufferRead {
                    output_index,
                    node: output.spec.node,
                    source,
                }
            })?;

            outputs.push(PlanOutputValue {
                node: output.spec.node,
                tensor_type: output.spec.tensor_type.clone(),
                values: decode_le(&bytes, output_index)?,
            });
        }

        Ok(PlanOutputs { values: outputs })
    }

    fn allocate(&self, bytes: usize) -> Result<B::Buffer, PlanExecutionError> {
        self.backend
            .allocate(bytes, REFERENCE_ALIGNMENT, REFERENCE_MEMORY_SPACE)
            .map_err(|source| PlanExecutionError::BufferAllocation { bytes, source })
    }
}

// ---------------------------------------------------------------------------
// Preparation helpers
// ---------------------------------------------------------------------------

/// Tensor types recovered from the plan's kernel arguments.
///
/// A `LoweredPlan` carries no memory plan and no per-value type table: the only
/// place a tensor type appears is on a `KernelArgument`. Slot and external types
/// are therefore reconstructed from every argument occurrence, and every
/// occurrence must agree. That is sound because the memory planner reuses a slot
/// only for a value whose tensor type is *exactly* equal.
///
/// External types may additionally be *seeded* from caller-supplied hints before
/// the argument scan; the scan then confronts every occurrence with the seed
/// instead of replacing it.
struct ValueTypes {
    slots: Vec<Option<TensorType>>,
    externals: Vec<Option<TensorType>>,
}

fn collect_value_types(
    plan: &LoweredPlan,
    external_types: &[(NodeId, TensorType)],
) -> Result<ValueTypes, PlanPreparationError> {
    let mut highest_slot: Option<usize> = None;
    for instruction in plan.instructions()
    {
        for argument in &instruction.arguments
        {
            if let KernelArgumentSource::Buffer(slot) = argument.source
            {
                highest_slot = Some(highest_slot.map_or(slot.get(), |current| {
                    if slot.get() > current
                    {
                        slot.get()
                    }
                    else
                    {
                        current
                    }
                }));
            }
        }
    }

    let slot_count = highest_slot.map_or(0, |highest| highest.saturating_add(1));
    let mut slots: Vec<Option<TensorType>> = vec![None; slot_count];
    let mut externals: Vec<Option<TensorType>> = vec![None; plan.bindings().len()];

    // Seed the external table with the caller's hints, in the caller's order.
    // The order is irrelevant: every hint is placed at its binding's position,
    // and a repeated node is rejected rather than allowed to overwrite.
    let mut seeded = vec![false; plan.bindings().len()];
    for (node, tensor_type) in external_types
    {
        let index = plan
            .bindings()
            .iter()
            .position(|entry| entry.node == *node)
            .ok_or(PlanPreparationError::UnknownExternalTypeNode { node: *node })?;

        if tensor_type.dtype != DType::F32
        {
            return Err(PlanPreparationError::UnsupportedDType { node: *node });
        }

        let marker = seeded
            .get_mut(index)
            .ok_or(PlanPreparationError::UnknownExternalTypeNode { node: *node })?;
        if *marker
        {
            return Err(PlanPreparationError::DuplicateExternalType { node: *node });
        }
        *marker = true;

        let entry = externals
            .get_mut(index)
            .ok_or(PlanPreparationError::UnknownExternalTypeNode { node: *node })?;
        *entry = Some(tensor_type.clone());
    }

    for (dispatch_index, instruction) in plan.instructions().iter().enumerate()
    {
        for (position, argument) in instruction.arguments.iter().enumerate()
        {
            match argument.source
            {
                KernelArgumentSource::Buffer(slot) =>
                {
                    let entry = slots
                        .get_mut(slot.get())
                        .ok_or(PlanPreparationError::NonDenseBufferSlot { slot })?;
                    record_type(entry, &argument.tensor_type)
                        .map_err(|()| PlanPreparationError::InconsistentSlotType { slot })?;
                },
                KernelArgumentSource::ExternalInput(binding)
                | KernelArgumentSource::ExternalConstant(binding) =>
                {
                    let declared =
                        plan.binding(binding)
                            .ok_or(PlanPreparationError::UnknownBindingId {
                                dispatch_index,
                                binding,
                            })?;

                    let expected_kind = match argument.source
                    {
                        KernelArgumentSource::ExternalInput(_) => ExternalValueKind::Input,
                        _ => ExternalValueKind::Constant,
                    };
                    if declared.kind != expected_kind
                    {
                        return Err(PlanPreparationError::BindingKindMismatch {
                            dispatch_index,
                            binding,
                        });
                    }

                    let entry = externals.get_mut(binding.get() as usize).ok_or(
                        PlanPreparationError::UnknownBindingId {
                            dispatch_index,
                            binding,
                        },
                    )?;

                    match entry
                    {
                        Some(existing) if *existing == argument.tensor_type =>
                        {},
                        Some(existing) =>
                        {
                            // A seeded entry means the caller's hint disagrees
                            // with the plan; an inferred one means the plan
                            // disagrees with itself. Two different faults, two
                            // different diagnostics.
                            if matches!(seeded.get(binding.get() as usize), Some(true))
                            {
                                return Err(PlanPreparationError::ExternalTypeContradiction {
                                    binding,
                                    node: declared.node,
                                    expected: existing.clone(),
                                    actual: argument.tensor_type.clone(),
                                });
                            }
                            return Err(PlanPreparationError::InconsistentExternalType { binding });
                        },
                        None => *entry = Some(argument.tensor_type.clone()),
                    }
                },
                // `KernelArgumentSource` is `#[non_exhaustive]`: a future source
                // kind is rejected, never guessed at.
                _ =>
                {
                    return Err(PlanPreparationError::UnsupportedArgumentSource {
                        dispatch_index,
                        position,
                    });
                },
            }
        }
    }

    Ok(ValueTypes { slots, externals })
}

fn record_type(entry: &mut Option<TensorType>, observed: &TensorType) -> Result<(), ()> {
    match entry
    {
        Some(existing) if existing == observed => Ok(()),
        Some(_) => Err(()),
        None =>
        {
            *entry = Some(observed.clone());
            Ok(())
        },
    }
}

/// Validates every dispatch and resolves its arguments to physical targets.
fn resolve_dispatches(
    plan: &LoweredPlan,
    types: &ValueTypes,
) -> Result<Vec<PreparedDispatch>, PlanPreparationError> {
    // Kernel identifiers must be their own position, so a `Vec` can index them.
    for (position, kernel) in plan.kernels().iter().enumerate()
    {
        if kernel.id().get() as usize != position
        {
            return Err(PlanPreparationError::NonDenseKernelId {
                position,
                found: kernel.id(),
            });
        }
    }

    let mut defined_slots = vec![false; types.slots.len()];
    let mut dispatches = Vec::with_capacity(plan.instructions().len());

    for (dispatch_index, instruction) in plan.instructions().iter().enumerate()
    {
        let kernel =
            plan.kernel(instruction.kernel)
                .ok_or(PlanPreparationError::UnknownKernelId {
                    dispatch_index,
                    kernel: instruction.kernel,
                })?;

        let expected_arguments = kernel.operands().len().saturating_add(1);
        if instruction.arguments.len() != expected_arguments
        {
            return Err(PlanPreparationError::ArgumentCountMismatch {
                dispatch_index,
                expected: expected_arguments,
                actual: instruction.arguments.len(),
            });
        }

        let mut writes = 0usize;
        let mut arguments = Vec::with_capacity(instruction.arguments.len());

        for (position, argument) in instruction.arguments.iter().enumerate()
        {
            if argument.index != index_as_u32(position)
            {
                return Err(PlanPreparationError::ArgumentIndexMismatch {
                    dispatch_index,
                    position,
                    found: argument.index,
                });
            }

            if argument.tensor_type.dtype != DType::F32
            {
                return Err(PlanPreparationError::UnsupportedDType {
                    node: instruction.node,
                });
            }

            // Narrow the non-exhaustive access once, here, so nothing downstream
            // has to guess about an unknown variant.
            let access = match argument.access
            {
                KernelArgumentAccess::Read => PreparedAccess::Read,
                KernelArgumentAccess::Write => PreparedAccess::Write,
                _ =>
                {
                    return Err(PlanPreparationError::UnsupportedArgumentAccess {
                        dispatch_index,
                        position,
                    });
                },
            };

            let signature = match access
            {
                PreparedAccess::Read => kernel.operands().get(position),
                PreparedAccess::Write => Some(kernel.result()),
            };
            if signature != Some(&argument.tensor_type)
            {
                return Err(PlanPreparationError::ArgumentTypeMismatch {
                    dispatch_index,
                    position,
                });
            }

            let target = match argument.source
            {
                KernelArgumentSource::Buffer(slot) => PhysicalTarget::Slot(slot.get()),
                KernelArgumentSource::ExternalInput(binding)
                | KernelArgumentSource::ExternalConstant(binding) =>
                {
                    if access == PreparedAccess::Write
                    {
                        return Err(PlanPreparationError::WriteToExternalValue {
                            dispatch_index,
                            binding,
                        });
                    }
                    PhysicalTarget::External(binding.get() as usize)
                },
                _ =>
                {
                    return Err(PlanPreparationError::UnsupportedArgumentSource {
                        dispatch_index,
                        position,
                    });
                },
            };

            match access
            {
                PreparedAccess::Read =>
                {
                    if let KernelArgumentSource::Buffer(slot) = argument.source
                    {
                        let defined = defined_slots
                            .get(slot.get())
                            .copied()
                            .ok_or(PlanPreparationError::NonDenseBufferSlot { slot })?;
                        if !defined
                        {
                            return Err(PlanPreparationError::ReadBeforeDefinition {
                                dispatch_index,
                                slot,
                            });
                        }
                    }
                },
                PreparedAccess::Write =>
                {
                    writes = writes.saturating_add(1);
                    if writes > 1
                    {
                        return Err(PlanPreparationError::MultipleWriteArguments {
                            dispatch_index,
                        });
                    }
                    if position + 1 != instruction.arguments.len()
                    {
                        return Err(PlanPreparationError::WriteArgumentNotLast {
                            dispatch_index,
                            position,
                        });
                    }
                },
            }

            let elements = element_count(&argument.tensor_type)?;
            arguments.push(PreparedArgument {
                binding_slot: argument.index,
                target,
                length_bytes: byte_size(elements)?,
                access,
            });
        }

        if writes == 0
        {
            return Err(PlanPreparationError::MissingWriteArgument { dispatch_index });
        }

        let result_elements = element_count(kernel.result())?;
        if instruction.dispatch.elements != result_elements
        {
            return Err(PlanPreparationError::DispatchExtentMismatch {
                dispatch_index,
                expected: result_elements,
                actual: instruction.dispatch.elements,
            });
        }

        // The write argument was validated as the last one above.
        if let Some(last) = instruction.arguments.last()
        {
            if let KernelArgumentSource::Buffer(slot) = last.source
            {
                if let Some(entry) = defined_slots.get_mut(slot.get())
                {
                    *entry = true;
                }
            }
        }

        dispatches.push(PreparedDispatch {
            node: instruction.node,
            kernel: instruction.kernel,
            arguments,
        });
    }

    Ok(dispatches)
}

fn build_slot_specifications(
    types: &ValueTypes,
    max_buffer_bytes: Option<usize>,
) -> Result<(Vec<BufferSlotSpecification>, Vec<usize>), PlanPreparationError> {
    let mut specifications = Vec::with_capacity(types.slots.len());
    let mut sizes = Vec::with_capacity(types.slots.len());

    for (index, entry) in types.slots.iter().enumerate()
    {
        let slot = BufferSlot::new(index);
        let tensor_type = entry
            .clone()
            .ok_or(PlanPreparationError::NonDenseBufferSlot { slot })?;

        let bytes = byte_size(element_count(&tensor_type)?)?;
        check_backend_limit(bytes, max_buffer_bytes)?;

        specifications.push(BufferSlotSpecification { slot, tensor_type });
        sizes.push(bytes);
    }

    Ok((specifications, sizes))
}

fn build_external_specifications(
    plan: &LoweredPlan,
    types: &ValueTypes,
    max_buffer_bytes: Option<usize>,
) -> Result<(Vec<ExternalValueSpec>, Vec<usize>), PlanPreparationError> {
    let mut specifications = Vec::with_capacity(plan.bindings().len());
    let mut sizes = Vec::with_capacity(plan.bindings().len());

    for (index, declared) in plan.bindings().iter().enumerate()
    {
        let tensor_type = types.externals.get(index).and_then(Clone::clone).ok_or(
            PlanPreparationError::UndeterminedValueType {
                node: declared.node,
            },
        )?;

        // Every external type reaching this point is already known to be `F32`:
        // an inferred one was checked by `resolve_dispatches`, a seeded one by
        // `collect_value_types`. No third source exists.
        let bytes = byte_size(element_count(&tensor_type)?)?;
        check_backend_limit(bytes, max_buffer_bytes)?;

        specifications.push(ExternalValueSpec {
            binding: LogicalBindingId::new(index_as_u32(index)),
            node: declared.node,
            kind: declared.kind,
            tensor_type,
        });
        sizes.push(bytes);
    }

    Ok((specifications, sizes))
}

fn resolve_outputs(
    plan: &LoweredPlan,
    types: &ValueTypes,
) -> Result<Vec<PreparedOutput>, PlanPreparationError> {
    let mut outputs = Vec::with_capacity(plan.outputs().len());

    for (output_index, output) in plan.outputs().iter().enumerate()
    {
        let (target, tensor_type) = match output.source
        {
            KernelArgumentSource::Buffer(slot) =>
            {
                let tensor_type = types.slots.get(slot.get()).and_then(Clone::clone).ok_or(
                    PlanPreparationError::UnknownOutputSource {
                        output_index,
                        node: output.node,
                    },
                )?;
                (PhysicalTarget::Slot(slot.get()), tensor_type)
            },
            KernelArgumentSource::ExternalInput(binding)
            | KernelArgumentSource::ExternalConstant(binding) =>
            {
                let index = binding.get() as usize;
                if index >= plan.bindings().len()
                {
                    return Err(PlanPreparationError::UnknownOutputSource {
                        output_index,
                        node: output.node,
                    });
                }

                let tensor_type = types
                    .externals
                    .get(index)
                    .and_then(Clone::clone)
                    .ok_or(PlanPreparationError::UndeterminedValueType { node: output.node })?;
                (PhysicalTarget::External(index), tensor_type)
            },
            _ =>
            {
                return Err(PlanPreparationError::UnknownOutputSource {
                    output_index,
                    node: output.node,
                });
            },
        };

        let length_bytes = byte_size(element_count(&tensor_type)?)?;
        outputs.push(PreparedOutput {
            spec: PlanOutputSpec {
                node: output.node,
                tensor_type,
            },
            target,
            length_bytes,
        });
    }

    Ok(outputs)
}

/// Generates every artefact once, rejects `Exp`/`Log`, then compiles each
/// logical kernel exactly once.
///
/// `ReferenceKernelGenerator::generate` is called a single time for the whole
/// plan — never once per kernel — and the resulting artefacts are walked in
/// `LogicalKernelId` order. Several dispatches naming the same kernel therefore
/// share one compiled kernel, because the lowerer already deduplicated them.
fn compile_kernels<B: ComputeBackend>(
    backend: &B,
    plan: &LoweredPlan,
) -> Result<Vec<B::Kernel>, PlanPreparationError> {
    let generated = ReferenceKernelGenerator::new()
        .generate(plan)
        .map_err(|source| PlanPreparationError::KernelGeneration { source })?;

    let artifacts = generated.artifacts();
    if artifacts.len() != plan.kernels().len()
    {
        return Err(PlanPreparationError::KernelArtifactMismatch {
            expected: plan.kernels().len(),
            actual: artifacts.len(),
        });
    }

    let mut kernels = Vec::with_capacity(artifacts.len());
    for (position, artifact) in artifacts.iter().enumerate()
    {
        let kernel = artifact.kernel_id();
        if kernel.get() as usize != position
        {
            return Err(PlanPreparationError::NonDenseKernelId {
                position,
                found: kernel,
            });
        }

        // Rejected here, by the runtime, before any physical compilation: the
        // executable opcode set is this runtime's contract, not the backend's.
        // A backend that happened to accept `Exp` or `Log` must not silently
        // widen what a Reference plan can do.
        let opcode = artifact.opcode();
        if matches!(opcode, ReferenceOpcode::Exp | ReferenceOpcode::Log)
        {
            return Err(PlanPreparationError::DeterministicMathUnavailable { kernel, opcode });
        }

        let module = artifact
            .to_kernel_module()
            .map_err(|source| PlanPreparationError::KernelModule { kernel, source })?;

        kernels.push(
            backend
                .compile(&module)
                .map_err(|source| PlanPreparationError::KernelCompilation { kernel, source })?,
        );
    }

    Ok(kernels)
}

// ---------------------------------------------------------------------------
// Execution helpers
// ---------------------------------------------------------------------------

/// Validates the supplied external values and orders them by binding.
///
/// Runs to completion before any allocation, so a rejected set of values costs
/// nothing and touches no backend state.
fn resolve_external_values<'a, K>(
    prepared: &PreparedReferencePlan<K>,
    values: &PlanExternalValues<'a>,
) -> Result<Vec<&'a [f32]>, PlanExecutionError> {
    let mut supplied: Vec<Option<&'a [f32]>> = vec![None; prepared.externals.len()];

    for &(binding, value) in values.entries()
    {
        let entry = supplied
            .get_mut(binding.get() as usize)
            .ok_or(PlanExecutionError::UnexpectedExternalValue { binding })?;

        if entry.is_some()
        {
            return Err(PlanExecutionError::DuplicateExternalValue { binding });
        }
        *entry = Some(value);
    }

    let mut resolved = Vec::with_capacity(prepared.externals.len());
    for (index, entry) in supplied.into_iter().enumerate()
    {
        let binding = LogicalBindingId::new(index_as_u32(index));
        let value = entry.ok_or(PlanExecutionError::MissingExternalValue { binding })?;

        let expected = prepared
            .external_sizes
            .get(index)
            .copied()
            .unwrap_or_default()
            / F32_BYTES;
        if value.len() != expected
        {
            return Err(PlanExecutionError::ExternalValueLengthMismatch {
                binding,
                expected,
                actual: value.len(),
            });
        }

        resolved.push(value);
    }

    Ok(resolved)
}

fn encode_le(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * F32_BYTES);
    for value in values
    {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn decode_le(bytes: &[u8], output_index: usize) -> Result<Vec<f32>, PlanExecutionError> {
    let (groups, remainder) = bytes.as_chunks::<F32_BYTES>();
    if !remainder.is_empty()
    {
        return Err(PlanExecutionError::MalformedOutputBytes {
            output_index,
            length_bytes: bytes.len(),
        });
    }

    Ok(groups
        .iter()
        .map(|group| f32::from_le_bytes(*group))
        .collect())
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn element_count(tensor_type: &TensorType) -> Result<usize, PlanPreparationError> {
    tensor_type
        .shape
        .checked_num_elements()
        .map_err(|_| PlanPreparationError::SizeOverflow { elements: 0 })
}

fn byte_size(elements: usize) -> Result<usize, PlanPreparationError> {
    elements
        .checked_mul(F32_BYTES)
        .ok_or(PlanPreparationError::SizeOverflow { elements })
}

fn check_backend_limit(
    bytes: usize,
    max_buffer_bytes: Option<usize>,
) -> Result<(), PlanPreparationError> {
    match max_buffer_bytes
    {
        Some(limit) if bytes > limit =>
        {
            Err(PlanPreparationError::BufferExceedsBackendLimit { bytes, limit })
        },
        _ => Ok(()),
    }
}

/// Positions are bounded by plan sizes, which the lowerer already caps below
/// `u32::MAX`; the saturating conversion therefore never loses information and
/// keeps this helper free of a `Result`.
fn index_as_u32(index: usize) -> u32 {
    u32::try_from(index).unwrap_or(u32::MAX)
}

#[cfg(test)]
pub(crate) mod test_backend {
    //! A `ComputeBackend` that records how it is driven.
    //!
    //! The real CPU path is covered end to end in this crate's integration
    //! tests. What a real backend cannot show is *how* a runtime drives the
    //! contract: how many times `compile` is called, in which order buffers are
    //! allocated and written, that every launch is followed by a wait, that
    //! `synchronize` happens once after the last dispatch, and that a failure
    //! stops everything. This backend records those calls and can fail a chosen
    //! dispatch on demand.
    //!
    //! It is `pub(crate)` so the graph-session tests reuse it instead of
    //! duplicating it, and `#[cfg(test)]` so it never becomes public API. It
    //! computes nothing: only ordering and bookkeeping are observable.

    use super::*;

    use core::cell::{Cell, RefCell};

    use scirust_compute::{
        ComputeError, ComputeResult, DeviceCapabilities, DeviceId, KernelFormat, KernelModule,
    };

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) enum Call {
        Allocate { bytes: usize },
        Write { bytes: usize },
        Compile { entry_point: String },
        CreateStream,
        Launch { entry_point: String },
        Wait,
        Synchronize,
        Read { bytes: usize },
    }

    #[derive(Debug)]
    pub(crate) struct RecordingBuffer {
        bytes: RefCell<Vec<u8>>,
    }

    #[derive(Debug)]
    pub(crate) struct RecordingBackend {
        capabilities: DeviceCapabilities,
        calls: RefCell<Vec<Call>>,
        launches: Cell<usize>,
        fail_launch_at: Cell<Option<usize>>,
    }

    impl RecordingBackend {
        pub(crate) fn new() -> Self {
            Self::with_capabilities(DeviceCapabilities {
                device: DeviceId::cpu(),
                name: "recording".to_string(),
                supported_dtypes: vec![DType::F32],
                max_buffer_bytes: None,
                max_workgroup_size: [1, 1, 1],
                supports_async_execution: false,
            })
        }

        pub(crate) fn with_capabilities(capabilities: DeviceCapabilities) -> Self {
            Self {
                capabilities,
                calls: RefCell::new(Vec::new()),
                launches: Cell::new(0),
                fail_launch_at: Cell::new(None),
            }
        }

        pub(crate) fn fail_launch_at(self, dispatch_index: usize) -> Self {
            self.fail_launch_at.set(Some(dispatch_index));
            self
        }

        /// Stops injecting launch failures, so the same backend can prove that a
        /// prepared plan or session survives one.
        pub(crate) fn clear_launch_failure(&self) {
            self.fail_launch_at.set(None);
        }

        fn record(&self, call: Call) {
            self.calls.borrow_mut().push(call);
        }

        pub(crate) fn calls(&self) -> Vec<Call> {
            self.calls.borrow().clone()
        }

        pub(crate) fn count<F: Fn(&Call) -> bool>(&self, predicate: F) -> usize {
            self.calls
                .borrow()
                .iter()
                .filter(|call| predicate(call))
                .count()
        }
    }

    impl ComputeBackend for RecordingBackend {
        type Buffer = RecordingBuffer;
        type Kernel = String;
        type Stream = ();
        type Event = ();

        fn capabilities(&self) -> &DeviceCapabilities {
            &self.capabilities
        }

        fn allocate(
            &self,
            bytes: usize,
            _alignment: usize,
            _memory_space: MemorySpace,
        ) -> ComputeResult<Self::Buffer> {
            self.record(Call::Allocate { bytes });
            Ok(RecordingBuffer {
                bytes: RefCell::new(vec![0u8; bytes]),
            })
        }

        fn write(
            &self,
            destination: &Self::Buffer,
            offset_bytes: usize,
            data: &[u8],
        ) -> ComputeResult<()> {
            self.record(Call::Write { bytes: data.len() });
            let mut storage = destination.bytes.borrow_mut();
            let end = offset_bytes + data.len();
            let range = storage
                .get_mut(offset_bytes..end)
                .ok_or(ComputeError::InvalidArgument("write out of bounds"))?;
            range.copy_from_slice(data);
            Ok(())
        }

        fn read(
            &self,
            source: &Self::Buffer,
            offset_bytes: usize,
            destination: &mut [u8],
        ) -> ComputeResult<()> {
            self.record(Call::Read {
                bytes: destination.len(),
            });
            let storage = source.bytes.borrow();
            let end = offset_bytes + destination.len();
            let range = storage
                .get(offset_bytes..end)
                .ok_or(ComputeError::InvalidArgument("read out of bounds"))?;
            destination.copy_from_slice(range);
            Ok(())
        }

        fn compile(&self, module: &KernelModule) -> ComputeResult<Self::Kernel> {
            if module.format != KernelFormat::Reference
            {
                return Err(ComputeError::Unsupported("reference modules only"));
            }
            self.record(Call::Compile {
                entry_point: module.entry_point.clone(),
            });
            Ok(module.entry_point.clone())
        }

        fn create_stream(&self) -> ComputeResult<Self::Stream> {
            self.record(Call::CreateStream);
            Ok(())
        }

        fn launch(
            &self,
            kernel: &Self::Kernel,
            _stream: &Self::Stream,
            _config: LaunchConfig,
            _bindings: &[BufferBinding<'_, Self::Buffer>],
        ) -> ComputeResult<Self::Event> {
            let index = self.launches.get();
            self.launches.set(index + 1);
            self.record(Call::Launch {
                entry_point: kernel.clone(),
            });

            if self.fail_launch_at.get() == Some(index)
            {
                return Err(ComputeError::Launch("injected failure".to_string()));
            }
            Ok(())
        }

        fn wait(&self, _event: &Self::Event) -> ComputeResult<()> {
            self.record(Call::Wait);
            Ok(())
        }

        fn synchronize(&self, _stream: &Self::Stream) -> ComputeResult<()> {
            self.record(Call::Synchronize);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests of the plan runtime, driven by the recording backend.

    use super::test_backend::{Call, RecordingBackend, RecordingBuffer};
    use super::*;

    use core::error::Error as _;

    use scirust_compute::{
        ComputeError, ComputeResult, DeviceCapabilities, DeviceId, KernelModule, Shape,
    };
    use scirust_tensor_compile::{CanonicalCompiler, ExternalBindings, KernelLowerer};
    use scirust_tensor_ir::{Graph, Operation, Scalar};

    // -----------------------------------------------------------------------
    // Fixtures
    // -----------------------------------------------------------------------

    fn f32_type(dims: Vec<usize>) -> TensorType {
        TensorType::new(DType::F32, Shape::new(dims))
    }

    fn lower(graph: &Graph) -> LoweredPlan {
        let plan = CanonicalCompiler::new()
            .compile(graph)
            .expect("valid graph");
        let bindings = ExternalBindings::derive(&plan);
        KernelLowerer::new()
            .lower(&plan, &bindings)
            .expect("valid lowering")
    }

    /// `x -> relu -> relu -> output`: two dispatches, one shared kernel.
    fn shared_kernel_plan() -> LoweredPlan {
        let ty = f32_type(vec![2]);
        let mut graph = Graph::new();
        let input = graph.add_input("x", ty.clone()).expect("input");
        let first = graph
            .add_node(Operation::Relu, vec![input], ty.clone())
            .expect("relu");
        let second = graph
            .add_node(Operation::Relu, vec![first], ty)
            .expect("relu");
        graph.set_outputs(vec![second]).expect("outputs");
        lower(&graph)
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[test]
    fn compile_is_called_once_per_logical_kernel_not_once_per_dispatch() {
        let plan = shared_kernel_plan();
        let runtime = ReferencePlanRuntime::new(RecordingBackend::new());
        let prepared = runtime.prepare(&plan).expect("preparable");

        assert_eq!(prepared.dispatch_count(), 2);
        assert_eq!(prepared.kernel_count(), 1);
        assert_eq!(
            runtime
                .backend()
                .count(|call| matches!(call, Call::Compile { .. })),
            1,
            "two dispatches sharing one logical kernel must compile it once"
        );
    }

    #[test]
    fn preparation_never_allocates_launches_or_synchronizes() {
        let plan = shared_kernel_plan();
        let runtime = ReferencePlanRuntime::new(RecordingBackend::new());
        runtime.prepare(&plan).expect("preparable");

        let calls = runtime.backend().calls();
        assert!(
            calls
                .iter()
                .all(|call| matches!(call, Call::Compile { .. })),
            "preparation must only compile, got {calls:?}"
        );
    }

    #[test]
    fn execution_follows_the_canonical_call_order() {
        let plan = shared_kernel_plan();
        let runtime = ReferencePlanRuntime::new(RecordingBackend::new());
        let prepared = runtime.prepare(&plan).expect("preparable");

        let input = [-1.0f32, 2.0];
        let mut values = PlanExternalValues::new();
        values.bind(LogicalBindingId::new(0), &input);
        runtime.execute(&prepared, &values).expect("executes");

        let calls = runtime.backend().calls();
        // One compile, then: slot buffers, external buffers, imports, stream,
        // launch/wait per dispatch, one synchronize, then the reads.
        let expected = vec![
            Call::Compile {
                entry_point: "scirust_reference_kernel_0".to_string(),
            },
            Call::Allocate { bytes: 8 },
            Call::Allocate { bytes: 8 },
            Call::Allocate { bytes: 8 },
            Call::Write { bytes: 8 },
            Call::CreateStream,
            Call::Launch {
                entry_point: "scirust_reference_kernel_0".to_string(),
            },
            Call::Wait,
            Call::Launch {
                entry_point: "scirust_reference_kernel_0".to_string(),
            },
            Call::Wait,
            Call::Synchronize,
            Call::Read { bytes: 8 },
        ];

        assert_eq!(calls, expected);
    }

    #[test]
    fn every_launch_is_followed_by_a_wait_and_one_synchronize_closes_the_run() {
        let plan = shared_kernel_plan();
        let runtime = ReferencePlanRuntime::new(RecordingBackend::new());
        let prepared = runtime.prepare(&plan).expect("preparable");

        let input = [1.0f32, 2.0];
        let mut values = PlanExternalValues::new();
        values.bind(LogicalBindingId::new(0), &input);
        runtime.execute(&prepared, &values).expect("executes");

        let backend = runtime.backend();
        let launches = backend.count(|call| matches!(call, Call::Launch { .. }));
        let waits = backend.count(|call| matches!(call, Call::Wait));

        assert_eq!(launches, prepared.dispatch_count());
        assert_eq!(waits, launches);
        assert_eq!(backend.count(|call| matches!(call, Call::Synchronize)), 1);
        assert_eq!(backend.count(|call| matches!(call, Call::CreateStream)), 1);

        // The synchronize is the last control call, after every launch/wait.
        let calls = backend.calls();
        let synchronize = calls
            .iter()
            .position(|call| matches!(call, Call::Synchronize))
            .expect("synchronize");
        let last_wait = calls
            .iter()
            .rposition(|call| matches!(call, Call::Wait))
            .expect("wait");
        assert!(synchronize > last_wait);
    }

    #[test]
    fn a_failing_dispatch_stops_the_plan_and_reads_nothing_back() {
        let plan = shared_kernel_plan();
        let runtime = ReferencePlanRuntime::new(RecordingBackend::new().fail_launch_at(1));
        let prepared = runtime.prepare(&plan).expect("preparable");

        let input = [1.0f32, 2.0];
        let mut values = PlanExternalValues::new();
        values.bind(LogicalBindingId::new(0), &input);

        let error = runtime
            .execute(&prepared, &values)
            .expect_err("the second dispatch fails");

        match &error
        {
            PlanExecutionError::KernelLaunch {
                dispatch_index,
                kernel,
                ..
            } =>
            {
                assert_eq!(*dispatch_index, 1);
                assert_eq!(kernel.get(), 0);
            },
            other => panic!("expected a launch failure, got {other:?}"),
        }

        // The backend error survives as a source rather than being flattened.
        assert!(error.source().is_some());

        let backend = runtime.backend();
        assert_eq!(backend.count(|call| matches!(call, Call::Launch { .. })), 2);
        assert_eq!(
            backend.count(|call| matches!(call, Call::Synchronize)),
            0,
            "a failed plan must not synchronize"
        );
        assert_eq!(
            backend.count(|call| matches!(call, Call::Read { .. })),
            0,
            "a failed plan must read no output back"
        );
    }

    #[test]
    fn a_prepared_plan_survives_a_failed_execution() {
        let plan = shared_kernel_plan();
        let runtime = ReferencePlanRuntime::new(RecordingBackend::new().fail_launch_at(0));
        let prepared = runtime.prepare(&plan).expect("preparable");

        let input = [1.0f32, 2.0];
        let mut values = PlanExternalValues::new();
        values.bind(LogicalBindingId::new(0), &input);

        assert!(runtime.execute(&prepared, &values).is_err());

        // Stop injecting failures; the same prepared plan runs.
        runtime.backend().clear_launch_failure();
        assert!(runtime.execute(&prepared, &values).is_ok());
    }

    #[test]
    fn a_buffer_exceeding_the_backend_limit_is_rejected_at_preparation() {
        let plan = shared_kernel_plan();
        let backend = RecordingBackend::with_capabilities(DeviceCapabilities {
            device: DeviceId::cpu(),
            name: "tiny".to_string(),
            supported_dtypes: vec![DType::F32],
            // Each buffer here is 8 bytes; 4 is below that.
            max_buffer_bytes: Some(4),
            max_workgroup_size: [1, 1, 1],
            supports_async_execution: false,
        });

        assert_eq!(
            ReferencePlanRuntime::new(backend).prepare(&plan).err(),
            Some(PlanPreparationError::BufferExceedsBackendLimit { bytes: 8, limit: 4 })
        );
    }

    #[test]
    fn a_backend_that_does_not_support_f32_is_rejected() {
        let plan = shared_kernel_plan();
        let backend = RecordingBackend::with_capabilities(DeviceCapabilities {
            device: DeviceId::cpu(),
            name: "no-f32".to_string(),
            supported_dtypes: vec![DType::I32],
            max_buffer_bytes: None,
            max_workgroup_size: [1, 1, 1],
            supports_async_execution: false,
        });

        assert_eq!(
            ReferencePlanRuntime::new(backend).prepare(&plan).err(),
            Some(PlanPreparationError::BackendDTypeUnsupported)
        );
    }

    #[test]
    fn a_backend_rejecting_a_module_reports_the_kernel_and_keeps_the_source() {
        /// Accepts nothing, so `compile` always fails.
        #[derive(Debug)]
        struct RefusingBackend {
            inner: RecordingBackend,
        }

        impl ComputeBackend for RefusingBackend {
            type Buffer = RecordingBuffer;
            type Kernel = String;
            type Stream = ();
            type Event = ();

            fn capabilities(&self) -> &DeviceCapabilities {
                self.inner.capabilities()
            }
            fn allocate(
                &self,
                bytes: usize,
                alignment: usize,
                memory_space: MemorySpace,
            ) -> ComputeResult<Self::Buffer> {
                self.inner.allocate(bytes, alignment, memory_space)
            }
            fn write(
                &self,
                destination: &Self::Buffer,
                offset_bytes: usize,
                data: &[u8],
            ) -> ComputeResult<()> {
                self.inner.write(destination, offset_bytes, data)
            }
            fn read(
                &self,
                source: &Self::Buffer,
                offset_bytes: usize,
                destination: &mut [u8],
            ) -> ComputeResult<()> {
                self.inner.read(source, offset_bytes, destination)
            }
            fn compile(&self, _module: &KernelModule) -> ComputeResult<Self::Kernel> {
                Err(ComputeError::Compilation("refused".to_string()))
            }
            fn create_stream(&self) -> ComputeResult<Self::Stream> {
                self.inner.create_stream()
            }
            fn launch(
                &self,
                kernel: &Self::Kernel,
                stream: &Self::Stream,
                config: LaunchConfig,
                bindings: &[BufferBinding<'_, Self::Buffer>],
            ) -> ComputeResult<Self::Event> {
                self.inner.launch(kernel, stream, config, bindings)
            }
            fn wait(&self, event: &Self::Event) -> ComputeResult<()> {
                self.inner.wait(event)
            }
            fn synchronize(&self, stream: &Self::Stream) -> ComputeResult<()> {
                self.inner.synchronize(stream)
            }
        }

        let plan = shared_kernel_plan();
        let runtime = ReferencePlanRuntime::new(RefusingBackend {
            inner: RecordingBackend::new(),
        });

        let error = runtime.prepare(&plan).expect_err("compilation is refused");
        match &error
        {
            PlanPreparationError::KernelCompilation { kernel, .. } =>
            {
                assert_eq!(kernel.get(), 0);
            },
            other => panic!("expected a compilation failure, got {other:?}"),
        }
        assert!(error.source().is_some());
    }

    #[test]
    fn exp_and_log_are_rejected_before_any_compilation() {
        for operation in [Operation::Exp, Operation::Log]
        {
            let ty = f32_type(vec![2]);
            let mut graph = Graph::new();
            let input = graph.add_input("x", ty.clone()).expect("input");
            let result = graph.add_node(operation, vec![input], ty).expect("node");
            graph.set_outputs(vec![result]).expect("outputs");
            let plan = lower(&graph);

            let runtime = ReferencePlanRuntime::new(RecordingBackend::new());
            let error = runtime.prepare(&plan).expect_err("must be rejected");

            assert!(matches!(
                error,
                PlanPreparationError::DeterministicMathUnavailable { .. }
            ));
            assert_eq!(
                runtime
                    .backend()
                    .count(|call| matches!(call, Call::Compile { .. })),
                0,
                "the runtime must reject Exp/Log before compiling anything"
            );
        }
    }

    #[test]
    fn external_values_are_validated_before_any_allocation() {
        let plan = shared_kernel_plan();
        let runtime = ReferencePlanRuntime::new(RecordingBackend::new());
        let prepared = runtime.prepare(&plan).expect("preparable");

        let wrong = [1.0f32];
        let mut values = PlanExternalValues::new();
        values.bind(LogicalBindingId::new(0), &wrong);

        assert!(runtime.execute(&prepared, &values).is_err());
        assert_eq!(
            runtime
                .backend()
                .count(|call| matches!(call, Call::Allocate { .. })),
            0,
            "a rejected value set must cost no allocation"
        );
    }

    #[test]
    fn scale_factors_travel_through_the_generated_kernel() {
        // A Scale kernel's factor lives in the artefact, not in a binding, so a
        // scale plan needs exactly one external value.
        let ty = f32_type(vec![2]);
        let mut graph = Graph::new();
        let input = graph.add_input("x", ty.clone()).expect("input");
        let scaled = graph
            .add_node(
                Operation::Scale {
                    factor: Scalar::f32(0.25),
                },
                vec![input],
                ty,
            )
            .expect("scale");
        graph.set_outputs(vec![scaled]).expect("outputs");
        let plan = lower(&graph);

        let runtime = ReferencePlanRuntime::new(RecordingBackend::new());
        let prepared = runtime.prepare(&plan).expect("preparable");

        assert_eq!(prepared.external_values().len(), 1);
        assert_eq!(prepared.kernel_count(), 1);
    }

    // -----------------------------------------------------------------------
    // External type hints
    // -----------------------------------------------------------------------

    /// `x -> output`: the input is the only output and never a kernel argument,
    /// so its tensor type appears nowhere in the lowered plan.
    fn untyped_external_plan() -> (LoweredPlan, NodeId, TensorType) {
        let ty = f32_type(vec![4]);
        let mut graph = Graph::new();
        let input = graph.add_input("x", ty.clone()).expect("input");
        graph.set_outputs(vec![input]).expect("outputs");
        (lower(&graph), input, ty)
    }

    #[test]
    fn prepare_still_rejects_an_external_whose_type_is_undeterminable() {
        let (plan, node, _) = untyped_external_plan();
        let runtime = ReferencePlanRuntime::new(RecordingBackend::new());

        assert_eq!(
            runtime.prepare(&plan).err(),
            Some(PlanPreparationError::UndeterminedValueType { node })
        );
        assert!(
            runtime.backend().calls().is_empty(),
            "a rejected plan must not reach the backend"
        );
    }

    #[test]
    fn an_external_type_hint_completes_missing_metadata() {
        let (plan, node, ty) = untyped_external_plan();
        let runtime = ReferencePlanRuntime::new(RecordingBackend::new());

        let prepared = runtime
            .prepare_with_external_types(&plan, &[(node, ty.clone())])
            .expect("hint completes the missing type");

        assert_eq!(prepared.kernel_count(), 0);
        assert_eq!(prepared.dispatch_count(), 0);
        assert_eq!(
            prepared.external_values(),
            &[ExternalValueSpec {
                binding: LogicalBindingId::new(0),
                node,
                kind: ExternalValueKind::Input,
                tensor_type: ty.clone(),
            }]
        );
        assert_eq!(
            prepared.outputs(),
            vec![PlanOutputSpec {
                node,
                tensor_type: ty
            }]
        );
        assert!(
            runtime.backend().calls().is_empty(),
            "a plan without kernels compiles nothing"
        );
    }

    #[test]
    fn an_unknown_hint_node_is_rejected() {
        let (plan, _, ty) = untyped_external_plan();
        let stranger = NodeId::new(9);

        assert_eq!(
            ReferencePlanRuntime::new(RecordingBackend::new())
                .prepare_with_external_types(&plan, &[(stranger, ty)])
                .err(),
            Some(PlanPreparationError::UnknownExternalTypeNode { node: stranger })
        );
    }

    #[test]
    fn a_duplicated_hint_is_rejected() {
        let (plan, node, ty) = untyped_external_plan();

        assert_eq!(
            ReferencePlanRuntime::new(RecordingBackend::new())
                .prepare_with_external_types(&plan, &[(node, ty.clone()), (node, ty)])
                .err(),
            Some(PlanPreparationError::DuplicateExternalType { node })
        );
    }

    #[test]
    fn a_non_f32_hint_is_rejected() {
        let (plan, node, _) = untyped_external_plan();
        let wrong = TensorType::new(DType::F64, Shape::new(vec![4]));

        assert_eq!(
            ReferencePlanRuntime::new(RecordingBackend::new())
                .prepare_with_external_types(&plan, &[(node, wrong)])
                .err(),
            Some(PlanPreparationError::UnsupportedDType { node })
        );
    }

    #[test]
    fn a_hint_contradicting_the_plan_is_rejected() {
        // Here the input *is* a kernel argument, so the plan states its type and
        // the hint can only agree or contradict.
        let plan = shared_kernel_plan();
        let node = NodeId::new(0);
        let observed = f32_type(vec![2]);
        let claimed = f32_type(vec![8]);

        assert_eq!(
            ReferencePlanRuntime::new(RecordingBackend::new())
                .prepare_with_external_types(&plan, &[(node, claimed.clone())])
                .err(),
            Some(PlanPreparationError::ExternalTypeContradiction {
                binding: LogicalBindingId::new(0),
                node,
                expected: claimed,
                actual: observed,
            })
        );
    }

    #[test]
    fn a_hint_agreeing_with_the_plan_changes_nothing() {
        let plan = shared_kernel_plan();
        let node = NodeId::new(0);
        let runtime = ReferencePlanRuntime::new(RecordingBackend::new());

        let hinted = runtime
            .prepare_with_external_types(&plan, &[(node, f32_type(vec![2]))])
            .expect("agreeing hint");
        let plain = runtime.prepare(&plan).expect("no hint");

        assert_eq!(hinted.external_values(), plain.external_values());
        assert_eq!(hinted.buffer_slots(), plain.buffer_slots());
        assert_eq!(hinted.outputs(), plain.outputs());
        assert_eq!(hinted.dispatch_count(), plain.dispatch_count());
        assert_eq!(hinted.kernel_count(), plain.kernel_count());
    }
}
