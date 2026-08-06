//! CPU execution of `KernelFormat::Reference` modules, behind the
//! `ComputeBackend` contract.
//!
//! # Binding ABI
//!
//! ```text
//! slot 0 .. operand_count-1   read operands, in the kernel's operand order
//! slot operand_count          the single written result
//! ```
//!
//! Exactly `operand_count + 1` bindings, no duplicate slot, no gap, no extra.
//! Slots are contiguous. **The order of the `bindings` slice is not
//! significant**: every binding is resolved by its slot number into a
//! position-indexed table, so the same set of bindings in any order behaves
//! identically. Nothing is sorted, and no ambiguity is silently resolved.
//!
//! # Access rules
//!
//! Operands must be [`BufferAccess::ReadOnly`], the result must be
//! [`BufferAccess::WriteOnly`]. [`BufferAccess::ReadWrite`] is rejected on both
//! sides in this version: the copy-in/copy-out executor never reads the result
//! and never writes an operand, so accepting `ReadWrite` would let a caller
//! declare an intent this backend does not honour. It is never *required*
//! either — an operand and the result may live in the same buffer as two
//! separate bindings, each with its own access. A later in-place CPU backend
//! could relax this.
//!
//! Note this is deliberately different from the WGPU adapter, where a storage
//! result *must* be `ReadWrite` because WGSL rejects write-only bindings. That
//! is a WGSL constraint, not a CPU one.
//!
//! # Byte ABI
//!
//! A Reference CPU buffer holds `f32` values as consecutive **little-endian**
//! 4-byte groups, converted with [`f32::from_le_bytes`] / [`f32::to_le_bytes`].
//!
//! Little-endian is explicit rather than native for two reasons: the Reference
//! wire format is itself strictly little-endian, so one endianness governs the
//! whole path; and a buffer's contents stay reproducible across hosts. This
//! differs from `cuda_compute_adapter`, which uses native-endian byte groups —
//! that is host-to-device staging on one machine, not a portable ABI. On x86_64
//! and ARM64 the two coincide; they diverge only on a big-endian target.
//!
//! `CpuBuffer` guarantees no alignment beyond one byte (`allocate` rejects any
//! alignment other than `1`), so reinterpreting its storage as `&[f32]` would be
//! unsound even with `unsafe`. Byte-wise conversion is the only correct option
//! here, not a precaution.
//!
//! # Transactional execution
//!
//! For a reference backend, correctness outranks throughput:
//!
//! 1. validate the launch configuration;
//! 2. resolve and validate **every** binding — slots, accesses, alignments,
//!    lengths and ranges — before touching any buffer contents;
//! 3. read and decode all operands into temporary `Vec<f32>`s, releasing each
//!    borrow as it completes;
//! 4. run the interpreter into a temporary `Vec<f32>`;
//! 5. only then encode and write the result range.
//!
//! Consequences: no partial write is possible; a failed launch leaves the result
//! buffer bit-for-bit unchanged; no mutable alias ever exists; and an operand
//! may share a `CpuBuffer` with the result — including an overlapping range —
//! because every read completes and every borrow is released before the first
//! write. A future optimised CPU backend can remove these copies without
//! changing the observable semantics.

extern crate alloc;

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use scirust_compute::{BufferAccess, BufferBinding, LaunchConfig};
use scirust_tensor_reference::{
    PreparedReferenceKernel, ReferenceExecutionError, ReferenceInterpreter,
};

use crate::compute_adapter::CpuBuffer;

/// Storage width of one `f32` in a Reference CPU buffer.
const F32_BYTES: usize = 4;

/// The only launch geometry this backend accepts.
///
/// The CPU adapter already advertises `max_workgroup_size == [1, 1, 1]`, the
/// interpreter processes a whole tensor in one serial call so there is no grid
/// to distribute, and the CPU has no shared memory. Any other configuration is
/// rejected rather than silently ignored.
const CANONICAL_GRID: [u32; 3] = [1, 1, 1];
const CANONICAL_BLOCK: [u32; 3] = [1, 1, 1];
const CANONICAL_SHARED_MEMORY_BYTES: u32 = 0;

/// A typed failure inside the CPU Reference launch path.
///
/// Kept typed all the way to the `ComputeBackend` boundary, where
/// `CpuComputeAdapter::launch` renders it into `ComputeError::Launch`. Rendering
/// happens once, at that boundary — never earlier, and never into an
/// undifferentiated message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CpuReferenceLaunchError {
    /// The binding count is not `operand_count + 1`.
    InvalidBindingCount { expected: usize, actual: usize },
    /// Two bindings claim the same slot.
    DuplicateBindingSlot { slot: u32 },
    /// A slot in `0..=operand_count` has no binding.
    ///
    /// Defensive: the count, range and duplicate checks together make a gap
    /// impossible — `operand_count + 1` bindings, each mapped to a distinct
    /// in-range slot, must cover every slot by the pigeonhole principle. Kept so
    /// a future relaxation of those checks fails loudly instead of reading an
    /// unset slot.
    MissingBindingSlot { slot: u32 },
    /// A binding names a slot outside `0..=operand_count`.
    UnexpectedBindingSlot { slot: u32, highest: u32 },
    /// An operand binding is not `ReadOnly`.
    InvalidOperandAccess { slot: u32, access: BufferAccess },
    /// The result binding is not `WriteOnly`.
    InvalidOutputAccess { slot: u32, access: BufferAccess },
    /// `offset_bytes` is not a multiple of 4.
    MisalignedOffset { slot: u32, offset_bytes: usize },
    /// `length_bytes` is not a multiple of 4.
    MisalignedLength { slot: u32, length_bytes: usize },
    /// `length_bytes` is aligned but does not equal `elements * 4`.
    BindingLengthMismatch {
        slot: u32,
        expected: usize,
        actual: usize,
    },
    /// `offset_bytes + length_bytes` overflows `usize`.
    BufferRangeOverflow { slot: u32 },
    /// The binding range extends past the end of its buffer.
    BufferOutOfBounds {
        slot: u32,
        end: usize,
        buffer_len: usize,
    },
    /// The buffer was already borrowed incompatibly.
    ///
    /// Reachable only if a caller holds a `CpuBuffer` borrow across a launch;
    /// reported rather than panicking, because this path never uses
    /// `RefCell::borrow`.
    BufferBorrowConflict { slot: u32 },
    /// `elements * 4` overflows `usize`.
    ///
    /// Defensive: the element count came from a prepared kernel that already
    /// converted it from the wire format through a checked conversion.
    ElementCountOverflow { slot: u32 },
    /// The launch geometry is not the canonical CPU Reference configuration.
    InvalidLaunchConfig {
        grid: [u32; 3],
        block: [u32; 3],
        shared_memory_bytes: u32,
    },
    /// A byte group could not be read as a 4-byte `f32`.
    ///
    /// Defensive: the range length is a validated multiple of 4, so
    /// `chunks_exact` always yields complete groups.
    MalformedElementBytes { slot: u32 },
    /// The interpreter rejected the kernel or the invocation.
    InvalidReferenceKernel(ReferenceExecutionError),
}

impl fmt::Display for CpuReferenceLaunchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidBindingCount { expected, actual } =>
            {
                write!(
                    formatter,
                    "reference kernel needs exactly {expected} binding(s) but received {actual}"
                )
            },
            Self::DuplicateBindingSlot { slot } =>
            {
                write!(formatter, "binding slot {slot} is bound more than once")
            },
            Self::MissingBindingSlot { slot } =>
            {
                write!(formatter, "binding slot {slot} has no binding")
            },
            Self::UnexpectedBindingSlot { slot, highest } =>
            {
                write!(
                    formatter,
                    "binding slot {slot} is outside the kernel's range 0..={highest}"
                )
            },
            Self::InvalidOperandAccess { slot, access } =>
            {
                write!(
                    formatter,
                    "operand slot {slot} declares {access:?}; reference operands must be ReadOnly"
                )
            },
            Self::InvalidOutputAccess { slot, access } =>
            {
                write!(
                    formatter,
                    "result slot {slot} declares {access:?}; the reference result must be WriteOnly"
                )
            },
            Self::MisalignedOffset { slot, offset_bytes } =>
            {
                write!(
                    formatter,
                    "slot {slot} offset {offset_bytes} is not a multiple of 4 bytes"
                )
            },
            Self::MisalignedLength { slot, length_bytes } =>
            {
                write!(
                    formatter,
                    "slot {slot} length {length_bytes} is not a multiple of 4 bytes"
                )
            },
            Self::BindingLengthMismatch {
                slot,
                expected,
                actual,
            } =>
            {
                write!(
                    formatter,
                    "slot {slot} must bind {expected} byte(s) but binds {actual}"
                )
            },
            Self::BufferRangeOverflow { slot } =>
            {
                write!(formatter, "slot {slot} binding range overflows usize")
            },
            Self::BufferOutOfBounds {
                slot,
                end,
                buffer_len,
            } =>
            {
                write!(
                    formatter,
                    "slot {slot} binding ends at {end} but its buffer holds {buffer_len} byte(s)"
                )
            },
            Self::BufferBorrowConflict { slot } =>
            {
                write!(formatter, "slot {slot} buffer is already borrowed")
            },
            Self::ElementCountOverflow { slot } =>
            {
                write!(formatter, "slot {slot} byte length overflows usize")
            },
            Self::InvalidLaunchConfig {
                grid,
                block,
                shared_memory_bytes,
            } =>
            {
                write!(
                    formatter,
                    "reference kernels require grid {CANONICAL_GRID:?}, block {CANONICAL_BLOCK:?} \
                     and {CANONICAL_SHARED_MEMORY_BYTES} shared byte(s), not grid {grid:?}, \
                     block {block:?} and {shared_memory_bytes} shared byte(s)"
                )
            },
            Self::MalformedElementBytes { slot } =>
            {
                write!(formatter, "slot {slot} holds an incomplete 4-byte element")
            },
            Self::InvalidReferenceKernel(error) =>
            {
                write!(formatter, "reference kernel execution failed: {error}")
            },
        }
    }
}

/// Runs one prepared Reference kernel over CPU buffer bindings.
///
/// Validates everything before reading, reads before executing, and writes only
/// after the interpreter has succeeded.
pub(crate) fn launch_reference_kernel(
    prepared: &PreparedReferenceKernel,
    config: LaunchConfig,
    bindings: &[BufferBinding<'_, CpuBuffer>],
) -> Result<(), CpuReferenceLaunchError> {
    validate_launch_config(config)?;

    let operand_lengths = prepared.operand_lengths();
    let operand_count = operand_lengths.len();
    let resolved = resolve_bindings(operand_count, bindings)?;

    // Validate every binding fully before any buffer contents are touched.
    for (slot_index, binding) in resolved.iter().enumerate()
    {
        let (elements, is_operand) = match operand_lengths.get(slot_index)
        {
            Some(&elements) => (elements, true),
            None => (prepared.result_length(), false),
        };

        let slot = binding.slot;
        if is_operand
        {
            if binding.access != BufferAccess::ReadOnly
            {
                return Err(CpuReferenceLaunchError::InvalidOperandAccess {
                    slot,
                    access: binding.access,
                });
            }
        }
        else if binding.access != BufferAccess::WriteOnly
        {
            return Err(CpuReferenceLaunchError::InvalidOutputAccess {
                slot,
                access: binding.access,
            });
        }

        validate_binding_range(binding, elements)?;
    }

    // Read every operand, releasing each borrow before the next step.
    let mut operand_storage = Vec::with_capacity(operand_count);
    for (slot_index, &elements) in operand_lengths.iter().enumerate()
    {
        let binding =
            resolved
                .get(slot_index)
                .ok_or(CpuReferenceLaunchError::MissingBindingSlot {
                    slot: slot_index_as_u32(slot_index),
                })?;
        operand_storage.push(read_operand(binding, elements)?);
    }

    let operands: Vec<&[f32]> = operand_storage.iter().map(Vec::as_slice).collect();

    let mut output = vec![0.0f32; prepared.result_length()];
    ReferenceInterpreter::new()
        .execute(prepared, &operands, &mut output)
        .map_err(CpuReferenceLaunchError::InvalidReferenceKernel)?;

    // Nothing has been written so far; only now does the result range change.
    let result_binding =
        resolved
            .get(operand_count)
            .ok_or(CpuReferenceLaunchError::MissingBindingSlot {
                slot: slot_index_as_u32(operand_count),
            })?;
    write_result(result_binding, &output)
}

fn validate_launch_config(config: LaunchConfig) -> Result<(), CpuReferenceLaunchError> {
    if config.grid != CANONICAL_GRID
        || config.block != CANONICAL_BLOCK
        || config.shared_memory_bytes != CANONICAL_SHARED_MEMORY_BYTES
    {
        return Err(CpuReferenceLaunchError::InvalidLaunchConfig {
            grid: config.grid,
            block: config.block,
            shared_memory_bytes: config.shared_memory_bytes,
        });
    }

    Ok(())
}

/// Maps bindings onto slots `0..=operand_count`, independently of slice order.
fn resolve_bindings<'a, 'buffer>(
    operand_count: usize,
    bindings: &'a [BufferBinding<'buffer, CpuBuffer>],
) -> Result<Vec<&'a BufferBinding<'buffer, CpuBuffer>>, CpuReferenceLaunchError> {
    let expected =
        operand_count
            .checked_add(1)
            .ok_or(CpuReferenceLaunchError::InvalidBindingCount {
                expected: usize::MAX,
                actual: bindings.len(),
            })?;

    if bindings.len() != expected
    {
        return Err(CpuReferenceLaunchError::InvalidBindingCount {
            expected,
            actual: bindings.len(),
        });
    }

    let highest = slot_index_as_u32(operand_count);
    let mut slots: Vec<Option<&'a BufferBinding<'buffer, CpuBuffer>>> = vec![None; expected];

    for binding in bindings
    {
        let index = usize::try_from(binding.slot).map_err(|_| {
            CpuReferenceLaunchError::UnexpectedBindingSlot {
                slot: binding.slot,
                highest,
            }
        })?;

        let entry = slots
            .get_mut(index)
            .ok_or(CpuReferenceLaunchError::UnexpectedBindingSlot {
                slot: binding.slot,
                highest,
            })?;

        if entry.is_some()
        {
            return Err(CpuReferenceLaunchError::DuplicateBindingSlot { slot: binding.slot });
        }

        *entry = Some(binding);
    }

    let mut resolved = Vec::with_capacity(expected);
    for (index, entry) in slots.into_iter().enumerate()
    {
        resolved.push(entry.ok_or(CpuReferenceLaunchError::MissingBindingSlot {
            slot: slot_index_as_u32(index),
        })?);
    }

    Ok(resolved)
}

/// Checks alignment, exact length and containment for one binding.
fn validate_binding_range(
    binding: &BufferBinding<'_, CpuBuffer>,
    elements: usize,
) -> Result<usize, CpuReferenceLaunchError> {
    let slot = binding.slot;

    if !binding.offset_bytes.is_multiple_of(F32_BYTES)
    {
        return Err(CpuReferenceLaunchError::MisalignedOffset {
            slot,
            offset_bytes: binding.offset_bytes,
        });
    }

    if !binding.length_bytes.is_multiple_of(F32_BYTES)
    {
        return Err(CpuReferenceLaunchError::MisalignedLength {
            slot,
            length_bytes: binding.length_bytes,
        });
    }

    let expected = elements
        .checked_mul(F32_BYTES)
        .ok_or(CpuReferenceLaunchError::ElementCountOverflow { slot })?;

    if binding.length_bytes != expected
    {
        return Err(CpuReferenceLaunchError::BindingLengthMismatch {
            slot,
            expected,
            actual: binding.length_bytes,
        });
    }

    let end = binding
        .offset_bytes
        .checked_add(binding.length_bytes)
        .ok_or(CpuReferenceLaunchError::BufferRangeOverflow { slot })?;

    let buffer_len = binding
        .buffer
        .bytes
        .try_borrow()
        .map_err(|_| CpuReferenceLaunchError::BufferBorrowConflict { slot })?
        .len();

    if end > buffer_len
    {
        return Err(CpuReferenceLaunchError::BufferOutOfBounds {
            slot,
            end,
            buffer_len,
        });
    }

    Ok(end)
}

/// Decodes one operand's little-endian byte range into `f32` values.
fn read_operand(
    binding: &BufferBinding<'_, CpuBuffer>,
    elements: usize,
) -> Result<Vec<f32>, CpuReferenceLaunchError> {
    let slot = binding.slot;
    let end = validate_binding_range(binding, elements)?;

    let bytes = binding
        .buffer
        .bytes
        .try_borrow()
        .map_err(|_| CpuReferenceLaunchError::BufferBorrowConflict { slot })?;

    let range =
        bytes
            .get(binding.offset_bytes..end)
            .ok_or(CpuReferenceLaunchError::BufferOutOfBounds {
                slot,
                end,
                buffer_len: bytes.len(),
            })?;

    let (groups, remainder) = range.as_chunks::<F32_BYTES>();
    if !remainder.is_empty()
    {
        return Err(CpuReferenceLaunchError::MalformedElementBytes { slot });
    }

    let mut values = Vec::with_capacity(elements);
    for group in groups
    {
        values.push(f32::from_le_bytes(*group));
    }

    Ok(values)
}

/// Encodes the interpreter's output into the result binding's byte range.
///
/// The only write in the whole launch path, and it happens only after every
/// preceding step has succeeded.
fn write_result(
    binding: &BufferBinding<'_, CpuBuffer>,
    values: &[f32],
) -> Result<(), CpuReferenceLaunchError> {
    let slot = binding.slot;
    let end = validate_binding_range(binding, values.len())?;

    let mut bytes = binding
        .buffer
        .bytes
        .try_borrow_mut()
        .map_err(|_| CpuReferenceLaunchError::BufferBorrowConflict { slot })?;
    let buffer_len = bytes.len();

    let range = bytes.get_mut(binding.offset_bytes..end).ok_or(
        CpuReferenceLaunchError::BufferOutOfBounds {
            slot,
            end,
            buffer_len,
        },
    )?;

    let (groups, remainder) = range.as_chunks_mut::<F32_BYTES>();
    if !remainder.is_empty()
    {
        return Err(CpuReferenceLaunchError::MalformedElementBytes { slot });
    }

    for (group, value) in groups.iter_mut().zip(values.iter())
    {
        *group = value.to_le_bytes();
    }

    Ok(())
}

/// Slot indices are bounded by the operand count, which a prepared kernel caps
/// far below `u32::MAX`; a saturating conversion therefore never loses
/// information here and keeps this helper free of a `Result`.
fn slot_index_as_u32(index: usize) -> u32 {
    u32::try_from(index).unwrap_or(u32::MAX)
}

/// Renders a launch failure for the `ComputeBackend` boundary.
pub(crate) fn describe(error: &CpuReferenceLaunchError) -> String {
    alloc::format!("{error}")
}

#[cfg(test)]
pub(crate) mod test_support {
    //! Builds genuine Reference artefacts through the real compilation
    //! pipeline, so tests exercise the bytes a production caller produces.

    use super::alloc::vec::Vec;

    use scirust_compute::{DType, KernelFormat, KernelModule, Shape};
    use scirust_tensor_compile::{CanonicalCompiler, ExternalBindings, KernelLowerer, LoweredPlan};
    use scirust_tensor_ir::{Graph, Operation, Scalar, TensorType};
    use scirust_tensor_reference::{ReferenceKernelArtifact, ReferenceKernelGenerator};

    pub(crate) fn f32_type(dims: Vec<usize>) -> TensorType {
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

    fn only_artifact(plan: &LoweredPlan) -> ReferenceKernelArtifact {
        let set = ReferenceKernelGenerator::new()
            .generate(plan)
            .expect("generatable plan");
        assert_eq!(set.len(), 1);
        set.artifacts()[0].clone()
    }

    pub(crate) fn unary_artifact(op: Operation, dims: Vec<usize>) -> ReferenceKernelArtifact {
        let ty = f32_type(dims);
        let mut graph = Graph::new();
        let input = graph.add_input("x", ty.clone()).expect("input");
        let result = graph.add_node(op, vec![input], ty).expect("node");
        graph.set_outputs(vec![result]).expect("outputs");
        only_artifact(&lower(&graph))
    }

    pub(crate) fn binary_artifact(op: Operation, dims: Vec<usize>) -> ReferenceKernelArtifact {
        let ty = f32_type(dims);
        let mut graph = Graph::new();
        let lhs = graph.add_input("lhs", ty.clone()).expect("lhs");
        let rhs = graph.add_input("rhs", ty.clone()).expect("rhs");
        let result = graph.add_node(op, vec![lhs, rhs], ty).expect("node");
        graph.set_outputs(vec![result]).expect("outputs");
        only_artifact(&lower(&graph))
    }

    pub(crate) fn scale_artifact(factor: f32, dims: Vec<usize>) -> ReferenceKernelArtifact {
        unary_artifact(
            Operation::Scale {
                factor: Scalar::f32(factor),
            },
            dims,
        )
    }

    pub(crate) fn reshape_artifact(
        input_dims: Vec<usize>,
        output_dims: Vec<usize>,
    ) -> ReferenceKernelArtifact {
        let mut graph = Graph::new();
        let input = graph.add_input("x", f32_type(input_dims)).expect("input");
        let result = graph
            .add_node(
                Operation::Reshape {
                    shape: Shape::new(output_dims.clone()),
                },
                vec![input],
                f32_type(output_dims),
            )
            .expect("node");
        graph.set_outputs(vec![result]).expect("outputs");
        only_artifact(&lower(&graph))
    }

    pub(crate) fn transpose_artifact(
        input_dims: Vec<usize>,
        output_dims: Vec<usize>,
        permutation: Vec<usize>,
    ) -> ReferenceKernelArtifact {
        let mut graph = Graph::new();
        let input = graph.add_input("x", f32_type(input_dims)).expect("input");
        let result = graph
            .add_node(
                Operation::Transpose { permutation },
                vec![input],
                f32_type(output_dims),
            )
            .expect("node");
        graph.set_outputs(vec![result]).expect("outputs");
        only_artifact(&lower(&graph))
    }

    pub(crate) fn module_of(artifact: &ReferenceKernelArtifact) -> KernelModule {
        artifact.to_kernel_module().expect("valid kernel module")
    }

    pub(crate) fn unary_module(op: Operation, dims: Vec<usize>) -> KernelModule {
        module_of(&unary_artifact(op, dims))
    }

    pub(crate) fn binary_module(op: Operation, dims: Vec<usize>) -> KernelModule {
        module_of(&binary_artifact(op, dims))
    }

    /// A Reference module whose payload is a valid artefact — the canonical
    /// "supported module" for conformance tests, replacing the arbitrary
    /// `b"reference"` bytes that predated real validation.
    pub(crate) fn canonical_reference_module() -> KernelModule {
        unary_module(Operation::Relu, vec![2, 2])
    }

    /// A syntactically valid `KernelModule` whose Reference payload is corrupt.
    pub(crate) fn corrupted_reference_module() -> KernelModule {
        let artifact = unary_artifact(Operation::Relu, vec![2, 2]);
        let mut code = artifact.encode();
        code[0] ^= 0xFF;
        KernelModule::new(KernelFormat::Reference, artifact.entry_point(), code)
            .expect("non-empty module")
    }
}
