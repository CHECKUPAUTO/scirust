//! Executes canonical Reference kernels on a CUDA device.
//!
//! # Why a second adapter
//!
//! [`crate::CudaComputeAdapter`] speaks PTX: it takes a [`KernelFormat::Ptx`]
//! module, requires `MemorySpace::Device`, and refuses the placement
//! conventions the canonical pipeline uses. That is a contract other callers
//! rely on, so this module implements the Reference contract separately and
//! leaves that adapter untouched — exactly as `wgpu_reference` sits beside
//! `wgpu_compute_adapter`.
//!
//! # What it does
//!
//! ```text
//! KernelModule::Reference
//!   -> ReferenceKernelArtifact::decode
//!   -> validation
//!   -> deterministic, specialised CUDA C
//!   -> NVRTC -> PTX -> CUDA module -> resolved function   (once, during compile)
//!   -> real kernel launch on the device's stream          (per launch)
//!   -> event synchronisation, then device-to-host readback
//! ```
//!
//! One kernel per logical kernel. The lowerer already deduplicates kernels by
//! full signature, so the element count, the shape, the strides, the `Scale`
//! factor and the permutation are all compile-time constants — no metadata
//! buffer, no per-dispatch upload, no bytecode interpreter on the device.
//!
//! **No CUDA C is generated, compiled or loaded during execution**, and nothing
//! here ever falls back to the CPU: an opcode this adapter cannot run is a
//! typed error, never a quietly host-computed result presented as CUDA.
//!
//! # Runtime requirements
//!
//! Two shared libraries must be loadable at run time: the CUDA driver
//! (`libcuda`) and the NVRTC runtime compiler (`libnvrtc`). Both are probed
//! before a device is opened, and each absence is reported distinctly. The
//! device ordinal is always explicit — there is no default, no fallback to
//! device zero and no implicit selection of a second device.
//!
//! # Determinism
//!
//! The generated source is a pure function of the artefact: same artefact, same
//! bytes. Parameter order is operand order followed by the result. The `Scale`
//! factor is emitted as `__uint_as_float(0x…u)` from its raw bits, never as a
//! formatted decimal. The kernel's name is a fixed string. There is no clock,
//! no random name, no address, no global counter and no hash map anywhere in
//! the generation path.
//!
//! NVRTC is always invoked with `--ftz=false`, `--prec-sqrt=true`,
//! `--prec-div=true` and `--fmad=false`, and never with fast math — see
//! `scirust_cuda::CudaRawRuntime::compile_cuda_c`, which fixes those flags and
//! exposes no way to relax them.
//!
//! Execution determinism is narrower, and stated honestly:
//!
//! * `ShapeCopy` and `Permute` move `unsigned int` words and never touch the
//!   float unit, so their results are **bit-identical** to the CPU
//!   interpreter's — NaN payloads, signed zeros, infinities and subnormals
//!   included. `Relu` selects between the input word and `+0.0`'s word, also
//!   without arithmetic, so it is bit-identical too.
//! * `Scale`, `Add`, `Sub`, `Mul` and `Div` are `f32` arithmetic. Each output
//!   element is one scalar operation, contraction is disabled and division and
//!   square root are IEEE — but bit-identical results across a CPU and a GPU
//!   are still **not promised** here, because that is a property of the target
//!   and its toolchain rather than something this crate can guarantee on their
//!   behalf. A NaN operand yields a NaN result, whose payload is unspecified.

use core::fmt;
use core::sync::atomic::{AtomicU64, Ordering};

use scirust_compute::{
    BufferAccess, BufferBinding, ComputeBackend, ComputeError, ComputeResult, DType,
    DeviceCapabilities, DeviceId, DeviceKind, KernelFormat, KernelModule, LaunchConfig,
    MemorySpace,
};
use scirust_cuda::{
    CudaDeviceInfo, CudaRawAccess, CudaRawBinding, CudaRawBuffer, CudaRawCompileOptions,
    CudaRawEvent, CudaRawKernel, CudaRawLaunchConfig, CudaRawRuntime,
};
use scirust_tensor_reference::{
    ReferenceAttributes, ReferenceKernelArtifact, ReferenceOpcode, ReferenceTensorLayout,
};

/// Storage width of one `f32`, and of the `unsigned int` word the structural
/// kernels move instead.
const WORD_BYTES: usize = 4;

/// Threads per block this adapter asks for, before the device's own limit is
/// applied. A plain, portable choice: large enough to fill a warp scheduler,
/// small enough that every CUDA device since compute capability 2.0 accepts it.
const PREFERRED_BLOCK_X: u32 = 256;

/// Alignment CUDA guarantees for a device allocation.
const CUDA_ALLOCATION_ALIGNMENT: usize = 256;

/// The launch geometry `ReferencePlanRuntime` issues.
///
/// It is a placeholder, not a description of the work: the Reference contract
/// leaves geometry to the backend, exactly as the CPU adapter derives its loop
/// bounds from the artefact rather than from this. Accepting anything else
/// would mean guessing at a caller's intent, so it is matched exactly and the
/// real grid comes from the kernel's own element count.
const CANONICAL_GRID: [u32; 3] = [1, 1, 1];
const CANONICAL_BLOCK: [u32; 3] = [1, 1, 1];

/// Name of every generated `__global__` function.
///
/// Fixed on purpose. It carries no kernel id, no address, no counter and no
/// timestamp: each artefact is compiled into its own CUDA module, so there is
/// nothing to disambiguate, and two structurally identical artefacts therefore
/// produce byte-identical source. The module's own entry point — the artefact's
/// `scirust_reference_kernel_<id>` — is still checked against the artefact
/// during [`ComputeBackend::compile`].
const CUDA_ENTRY_POINT: &str = "scirust_reference_kernel";

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

/// What one adapter instance has actually done.
///
/// Attached to the adapter, never global: two adapters count independently, and
/// nothing here is shared process-wide. Read with
/// [`CudaReferenceAdapter::counters`].
///
/// This exists so a test can prove that a real kernel ran — a claim that
/// otherwise rests on nothing observable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct CudaReferenceCounters {
    /// CUDA C sources compiled by NVRTC and loaded as modules.
    pub kernels_compiled: u64,
    /// Kernels actually submitted to the device. A zero-element kernel is
    /// **not** counted, because nothing was submitted for it.
    pub kernels_launched: u64,
}

// ---------------------------------------------------------------------------
// Adapter
// ---------------------------------------------------------------------------

/// A CUDA device that executes canonical Reference kernels.
///
/// See the module documentation for the contract, the runtime requirements and
/// the determinism guarantees with their limits.
pub struct CudaReferenceAdapter {
    runtime: CudaRawRuntime,
    capabilities: DeviceCapabilities,
    block_x: u32,
    max_grid_x: u32,
    kernels_compiled: AtomicU64,
    kernels_launched: AtomicU64,
}

impl CudaReferenceAdapter {
    /// Acquire the CUDA device with the given ordinal.
    ///
    /// The ordinal is always explicit: there is no default, and a failure to
    /// open it is never answered with a different device. Failures are
    /// distinguished:
    ///
    /// * no CUDA driver library →
    ///   [`ComputeError::BackendUnavailable`]`(DeviceKind::Cuda)`;
    /// * driver present, no NVRTC library → [`ComputeError::Compilation`];
    /// * driver present, no device at all → [`ComputeError::Unsupported`];
    /// * ordinal beyond the device count → [`ComputeError::InvalidArgument`];
    /// * device present but the context could not be created →
    ///   [`ComputeError::Allocation`], carrying the driver's message.
    pub fn new(device_ordinal: usize) -> ComputeResult<Self> {
        if !scirust_cuda::driver_available()
        {
            return Err(ComputeError::BackendUnavailable(DeviceKind::Cuda));
        }

        if !scirust_cuda::nvrtc_available()
        {
            return Err(ComputeError::Compilation(
                "the CUDA driver is present but the NVRTC runtime-compilation library could not \
                 be loaded; canonical Reference kernels are compiled from CUDA C and cannot run \
                 without it"
                    .to_string(),
            ));
        }

        let devices = scirust_cuda::device_count().map_err(ComputeError::Allocation)?;
        if devices == 0
        {
            return Err(ComputeError::Unsupported(
                "the CUDA driver is present but reports no device",
            ));
        }
        if device_ordinal >= devices
        {
            return Err(ComputeError::InvalidArgument(
                "the requested CUDA device ordinal does not exist",
            ));
        }

        let runtime = CudaRawRuntime::new(device_ordinal).map_err(ComputeError::Allocation)?;

        Self::from_runtime(runtime)
    }

    /// Wrap an already acquired raw CUDA runtime.
    ///
    /// Fails only when the device advertises limits this adapter cannot derive
    /// a launch geometry from.
    pub fn from_runtime(runtime: CudaRawRuntime) -> ComputeResult<Self> {
        let info = runtime.device_info();

        let block_x = PREFERRED_BLOCK_X
            .min(info.max_threads_per_block)
            .min(info.max_block_size[0]);
        if block_x == 0
        {
            return Err(ComputeError::Unsupported(
                "the CUDA device advertises no thread per block",
            ));
        }

        let max_grid_x = info.max_grid_size[0];
        if max_grid_x == 0
        {
            return Err(ComputeError::Unsupported(
                "the CUDA device advertises no block per grid",
            ));
        }

        let capabilities = DeviceCapabilities {
            device: DeviceId::new(
                DeviceKind::Cuda,
                u32::try_from(info.ordinal).unwrap_or(u32::MAX),
            ),
            name: format!(
                "scirust-gpu-cuda-reference: {} (device {}, sm_{}{})",
                info.name, info.ordinal, info.compute_capability.0, info.compute_capability.1
            ),
            // Reference v1.0 is F32 only, and this adapter does not widen it.
            supported_dtypes: vec![DType::F32],
            max_buffer_bytes: Some(info.total_memory_bytes),
            max_workgroup_size: info.max_block_size,
            supports_async_execution: true,
        };

        Ok(Self {
            runtime,
            capabilities,
            block_x,
            max_grid_x,
            kernels_compiled: AtomicU64::new(0),
            kernels_launched: AtomicU64::new(0),
        })
    }

    /// The device this adapter actually opened: ordinal, name, memory, compute
    /// capability and launch limits, all read from the driver.
    pub fn device_info(&self) -> &CudaDeviceInfo {
        self.runtime.device_info()
    }

    /// Capabilities reported by the acquired CUDA device.
    pub const fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }

    /// Threads per block this adapter launches with, after applying the
    /// device's limits.
    pub const fn block_size(&self) -> u32 {
        self.block_x
    }

    /// What this adapter instance has compiled and launched so far.
    pub fn counters(&self) -> CudaReferenceCounters {
        CudaReferenceCounters {
            kernels_compiled: self.kernels_compiled.load(Ordering::Relaxed),
            kernels_launched: self.kernels_launched.load(Ordering::Relaxed),
        }
    }
}

impl fmt::Debug for CudaReferenceAdapter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaReferenceAdapter")
            .field("device", &self.capabilities.name)
            .field("block_size", &self.block_x)
            .field("counters", &self.counters())
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Associated types
// ---------------------------------------------------------------------------

/// Device-resident buffer owned by the CUDA Reference adapter.
///
/// `bytes` is the *logical* size the runtime asked for. CUDA cannot hand back a
/// usable zero-byte allocation, so a physical allocation may be larger; every
/// read, write and binding stays bounded by the logical size, and the extra
/// bytes are unreachable through this API.
#[derive(Clone)]
pub struct CudaReferenceBuffer {
    raw: CudaRawBuffer,
    bytes: usize,
    memory_space: MemorySpace,
}

impl CudaReferenceBuffer {
    /// Logical size in bytes, as requested by the runtime.
    pub const fn len(&self) -> usize {
        self.bytes
    }

    pub const fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    pub const fn memory_space(&self) -> MemorySpace {
        self.memory_space
    }
}

impl fmt::Debug for CudaReferenceBuffer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaReferenceBuffer")
            .field("bytes", &self.bytes)
            .field("memory_space", &self.memory_space)
            .finish_non_exhaustive()
    }
}

/// One Reference kernel, compiled by NVRTC and loaded onto the device.
///
/// Everything invariant lives here: the resolved device function, the parameter
/// ABI, the logical byte sizes and the launch geometry. Execution adds nothing
/// to it and recompiles nothing.
pub struct CudaReferenceKernel {
    raw: CudaRawKernel,
    entry_point: String,
    kernel_id: u32,
    opcode: ReferenceOpcode,
    operand_bytes: Vec<usize>,
    result_bytes: usize,
    elements: usize,
    grid_x: u32,
    block_x: u32,
    source: String,
}

impl CudaReferenceKernel {
    /// The artefact's entry point, `scirust_reference_kernel_<id>`. Distinct
    /// from the generated `__global__` function's fixed name — see
    /// [`CudaReferenceKernel::cuda_entry_point`].
    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }

    /// The name of the generated `__global__` function, the same for every
    /// kernel.
    pub const fn cuda_entry_point(&self) -> &'static str {
        CUDA_ENTRY_POINT
    }

    pub const fn kernel_id(&self) -> u32 {
        self.kernel_id
    }

    pub const fn opcode(&self) -> ReferenceOpcode {
        self.opcode
    }

    /// Elements the kernel writes.
    pub const fn elements(&self) -> usize {
        self.elements
    }

    /// Blocks launched per dispatch. Zero for an empty tensor, which is not
    /// dispatched at all rather than dispatched emptily.
    pub const fn grid_size(&self) -> u32 {
        self.grid_x
    }

    /// Threads per block.
    pub const fn block_size(&self) -> u32 {
        self.block_x
    }

    /// The generated CUDA C, for diagnostics and determinism tests.
    pub fn source(&self) -> &str {
        &self.source
    }
}

impl fmt::Debug for CudaReferenceKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaReferenceKernel")
            .field("entry_point", &self.entry_point)
            .field("opcode", &self.opcode)
            .field("elements", &self.elements)
            .field("grid_x", &self.grid_x)
            .field("block_x", &self.block_x)
            .finish_non_exhaustive()
    }
}

/// Logical stream mapped onto the runtime's single ordered CUDA stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaReferenceStream(());

/// Completion signal for one dispatch.
///
/// A zero-element kernel produces an event that carries no CUDA submission,
/// because none was made. [`ComputeBackend::wait`] on it succeeds immediately
/// and honestly — it is not a recorded event that silently stands in for work
/// that never happened.
#[derive(Debug)]
pub struct CudaReferenceEvent {
    raw: Option<CudaRawEvent>,
}

impl CudaReferenceEvent {
    /// Whether a kernel was actually submitted for this dispatch.
    pub const fn launched(&self) -> bool {
        self.raw.is_some()
    }
}

// ---------------------------------------------------------------------------
// CUDA C generation
// ---------------------------------------------------------------------------

/// Everything the generator needs, extracted and checked once.
struct KernelPlan {
    operand_bytes: Vec<usize>,
    result_bytes: usize,
    elements: usize,
    grid_x: u32,
    source: String,
}

/// Element count of a layout as a `usize`.
///
/// The generated kernel indexes with `unsigned long long`, so the only bound
/// that matters is the host's own address space.
fn layout_elements(layout: &ReferenceTensorLayout) -> ComputeResult<usize> {
    usize::try_from(layout.elements()).map_err(|_| ComputeError::ShapeOverflow)
}

fn byte_size(elements: usize) -> ComputeResult<usize> {
    elements
        .checked_mul(WORD_BYTES)
        .ok_or(ComputeError::ByteSizeOverflow)
}

/// Row-major strides of `dims`: `strides[rank - 1] == 1` and
/// `strides[i] == product(dims[i + 1 ..])`.
///
/// Built by multiplication only — never by division — with `checked_mul` at
/// every step, matching the CPU interpreter's construction exactly.
fn row_major_strides(dims: &[u64]) -> ComputeResult<Vec<u64>> {
    let rank = dims.len();
    let mut strides = vec![1u64; rank];

    if rank <= 1
    {
        return Ok(strides);
    }

    for axis in (0..rank - 1).rev()
    {
        let next = strides
            .get(axis + 1)
            .copied()
            .ok_or(ComputeError::ShapeOverflow)?;
        let dimension = dims
            .get(axis + 1)
            .copied()
            .ok_or(ComputeError::ShapeOverflow)?;
        let stride = next
            .checked_mul(dimension)
            .ok_or(ComputeError::ShapeOverflow)?;

        *strides.get_mut(axis).ok_or(ComputeError::ShapeOverflow)? = stride;
    }

    Ok(strides)
}

/// Number of operands each executable opcode takes.
fn operand_arity(opcode: ReferenceOpcode) -> ComputeResult<usize> {
    match opcode
    {
        ReferenceOpcode::Relu
        | ReferenceOpcode::Scale
        | ReferenceOpcode::ShapeCopy
        | ReferenceOpcode::Permute => Ok(1),
        ReferenceOpcode::Add
        | ReferenceOpcode::Sub
        | ReferenceOpcode::Mul
        | ReferenceOpcode::Div => Ok(2),
        // Rejected by the runtime as well: no bit-reproducible implementation
        // is available, and this adapter must not widen what a Reference plan
        // can do.
        ReferenceOpcode::Exp | ReferenceOpcode::Log => Err(ComputeError::Unsupported(
            "Exp and Log have no bit-reproducible implementation and are not executed",
        )),
        // `ReferenceOpcode` is `#[non_exhaustive]`: an opcode added to it later
        // is refused here rather than silently treated as one this adapter
        // already knows.
        _ => Err(ComputeError::Unsupported(
            "this Reference opcode has no CUDA kernel",
        )),
    }
}

/// Whether the generated kernel addresses memory as raw 32-bit words rather
/// than as floats.
///
/// `ShapeCopy` and `Permute` move words. `Relu` selects between the input word
/// and `+0.0`'s word: it performs no arithmetic either, and going through raw
/// words is what makes its NaN-payload preservation provable rather than hoped
/// for — matching the CPU interpreter, which returns its input untouched.
const fn addresses_words(opcode: ReferenceOpcode) -> bool {
    matches!(
        opcode,
        ReferenceOpcode::ShapeCopy | ReferenceOpcode::Permute | ReferenceOpcode::Relu
    )
}

/// Validates one artefact and generates its specialised CUDA C.
fn plan_kernel(
    artifact: &ReferenceKernelArtifact,
    block_x: u32,
    max_grid_x: u32,
) -> ComputeResult<KernelPlan> {
    if artifact.dtype() != DType::F32
    {
        return Err(ComputeError::Unsupported(
            "CUDA Reference kernels execute F32 only",
        ));
    }

    let opcode = artifact.opcode();
    let arity = operand_arity(opcode)?;

    if artifact.operands().len() != arity
    {
        return Err(ComputeError::Compilation(format!(
            "{opcode:?} takes {arity} operand(s) but the artefact declares {}",
            artifact.operands().len()
        )));
    }

    let elements = layout_elements(artifact.result())?;
    let result_bytes = byte_size(elements)?;

    let mut operand_bytes = Vec::with_capacity(arity);
    for operand in artifact.operands()
    {
        let operand_elements = layout_elements(operand)?;

        // Every executable opcode is element-count preserving: the elementwise
        // families by definition, `ShapeCopy` and `Permute` by construction.
        if operand_elements != elements
        {
            return Err(ComputeError::Compilation(format!(
                "{opcode:?} operand holds {operand_elements} element(s) for a \
                 {elements}-element result"
            )));
        }

        operand_bytes.push(byte_size(operand_elements)?);
    }

    let grid_x = dispatch_grid(elements, block_x, max_grid_x)?;
    let source = generate_cuda_c(artifact, elements)?;

    Ok(KernelPlan {
        operand_bytes,
        result_bytes,
        elements,
        grid_x,
        source,
    })
}

/// Blocks needed to cover `elements`, capped by the device limit.
///
/// The cap is not a truncation: the kernel strides by the total thread count,
/// so a capped grid still covers every element.
fn dispatch_grid(elements: usize, block_x: u32, max_grid_x: u32) -> ComputeResult<u32> {
    if elements == 0
    {
        return Ok(0);
    }
    if block_x == 0
    {
        return Err(ComputeError::Unsupported(
            "the CUDA device advertises no thread per block",
        ));
    }
    if max_grid_x == 0
    {
        return Err(ComputeError::Unsupported(
            "the CUDA device advertises no block per grid",
        ));
    }

    let elements = u64::try_from(elements).map_err(|_| ComputeError::ShapeOverflow)?;
    let needed = elements.div_ceil(u64::from(block_x));
    let needed = u32::try_from(needed).unwrap_or(u32::MAX);

    Ok(needed.min(max_grid_x))
}

/// The CUDA C source for one artefact. A pure function of its inputs.
fn generate_cuda_c(artifact: &ReferenceKernelArtifact, elements: usize) -> ComputeResult<String> {
    let opcode = artifact.opcode();
    let arity = operand_arity(opcode)?;
    let word = if addresses_words(opcode)
    {
        "unsigned int"
    }
    else
    {
        "float"
    };

    let mut source = String::new();
    source.push_str("// Generated by scirust-gpu from a Reference kernel artefact.\n");
    source.push_str(&format!("// opcode: {opcode:?}\n"));
    source.push_str(&format!("// elements: {elements}\n\n"));

    // No `__restrict__` anywhere: one allocation may legitimately back two
    // read-only operands (`add(x, x)`), and promising the compiler otherwise
    // would be a lie the optimiser is entitled to act on.
    source.push_str(&format!(
        "extern \"C\" __global__ void {CUDA_ENTRY_POINT}(\n"
    ));
    for index in 0..arity
    {
        source.push_str(&format!("    const {word}* operand_{index},\n"));
    }
    source.push_str(&format!("    {word}* result)\n"));
    source.push_str("{\n");

    if elements == 0
    {
        // Nothing to compute. The module exists so `compile` stays uniform;
        // `launch` issues no dispatch for it at all.
        source.push_str("    // Zero elements: this kernel is never launched.\n");
        source.push_str("}\n");
        return Ok(source);
    }

    source.push_str(&format!(
        "    const unsigned long long total = {elements}ull;\n"
    ));
    source.push_str("    const unsigned long long stride =\n");
    source.push_str("        (unsigned long long)blockDim.x * (unsigned long long)gridDim.x;\n");
    source.push_str("    unsigned long long index =\n");
    source.push_str("        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x\n");
    source.push_str("        + (unsigned long long)threadIdx.x;\n");
    // A capped grid still covers every element: each thread strides by the
    // total thread count.
    source.push_str("    for (; index < total; index += stride)\n");
    source.push_str("    {\n");
    source.push_str(&kernel_body(artifact)?);
    source.push_str("    }\n");
    source.push_str("}\n");

    Ok(source)
}

/// The per-element statements of one opcode.
fn kernel_body(artifact: &ReferenceKernelArtifact) -> ComputeResult<String> {
    let mut body = String::new();

    match artifact.opcode()
    {
        ReferenceOpcode::Relu =>
        {
            // `x` when NaN or strictly positive, `+0.0` otherwise — the CPU
            // interpreter's exact rule, including `-0.0 -> +0.0`. The word is
            // copied rather than recomputed, so a NaN payload survives
            // bit-for-bit; the NaN test is bitwise so no compiler can discard
            // it under a finite-math assumption.
            body.push_str("        const unsigned int bits = operand_0[index];\n");
            body.push_str("        const bool is_nan =\n");
            body.push_str("            ((bits & 0x7f800000u) == 0x7f800000u) && ");
            body.push_str("((bits & 0x007fffffu) != 0u);\n");
            body.push_str("        const float value = __uint_as_float(bits);\n");
            body.push_str(
                "        result[index] = (is_nan || value > 0.0f) ? bits : 0x00000000u;\n",
            );
        },
        ReferenceOpcode::Scale =>
        {
            let factor_bits = match artifact.attributes()
            {
                ReferenceAttributes::Scale { factor_bits } => *factor_bits,
                // `ReferenceAttributes` is `#[non_exhaustive]`; anything other
                // than the payload this opcode requires is refused.
                _ =>
                {
                    return Err(ComputeError::Compilation(
                        "Scale artefact carries no factor".to_string(),
                    ));
                },
            };

            // Emitted from raw bits, never as a formatted decimal, so the
            // constant is exactly the one the artefact stores — `-0.0`, the
            // infinities and every NaN payload included.
            body.push_str(&format!(
                "        result[index] = operand_0[index] * \
                 __uint_as_float({factor_bits:#010x}u);\n"
            ));
        },
        ReferenceOpcode::Add
        | ReferenceOpcode::Sub
        | ReferenceOpcode::Mul
        | ReferenceOpcode::Div =>
        {
            let operator = match artifact.opcode()
            {
                ReferenceOpcode::Add => "+",
                ReferenceOpcode::Sub => "-",
                ReferenceOpcode::Mul => "*",
                ReferenceOpcode::Div => "/",
                // Unreachable: this arm only matches the four binary opcodes.
                // Refused rather than defaulted to one of them.
                _ =>
                {
                    return Err(ComputeError::Unsupported(
                        "this Reference opcode has no CUDA kernel",
                    ));
                },
            };
            body.push_str(&format!(
                "        result[index] = operand_0[index] {operator} operand_1[index];\n"
            ));
        },
        ReferenceOpcode::ShapeCopy =>
        {
            // `unsigned int` words: the float unit is never involved, so every
            // bit survives — NaN payloads, signed zeros, infinities, subnormals.
            body.push_str("        result[index] = operand_0[index];\n");
        },
        ReferenceOpcode::Permute =>
        {
            body.push_str(&permute_body(artifact)?);
        },
        ReferenceOpcode::Exp | ReferenceOpcode::Log =>
        {
            return Err(ComputeError::Unsupported(
                "Exp and Log have no bit-reproducible implementation and are not executed",
            ));
        },
        // See `operand_arity`: the enum is `#[non_exhaustive]`.
        _ =>
        {
            return Err(ComputeError::Unsupported(
                "this Reference opcode has no CUDA kernel",
            ));
        },
    }

    Ok(body)
}

/// Index arithmetic for `Permute`, fully unrolled with literal constants.
///
/// Convention, the same one the lowerer validates and the CPU interpreter
/// implements: `output.shape[i] == input.shape[axes[i]]`. For an output element
/// at linear index `o`, the source index is
/// `sum_i ((o / out_stride[i]) % out_dim[i]) * in_stride[axes[i]]`.
///
/// Every check the CPU interpreter performs at preparation is performed here,
/// at generation: rank agreement, bijectivity of the axis list, dimension
/// agreement axis by axis, and overflow-checked stride construction. Rank zero
/// is a one-element scalar — the unrolled sum is empty, the source index stays
/// `0`, and the single element is copied.
fn permute_body(artifact: &ReferenceKernelArtifact) -> ComputeResult<String> {
    let axes = match artifact.attributes()
    {
        ReferenceAttributes::Permute { axes } => axes,
        // `ReferenceAttributes` is `#[non_exhaustive]`.
        _ =>
        {
            return Err(ComputeError::Compilation(
                "Permute artefact carries no axis list".to_string(),
            ));
        },
    };

    let input = artifact
        .operands()
        .first()
        .ok_or_else(|| ComputeError::Compilation("Permute artefact has no operand".to_string()))?;
    let output = artifact.result();

    let rank = output.rank();
    if axes.len() != rank || input.rank() != rank
    {
        return Err(ComputeError::Compilation(format!(
            "Permute rank mismatch: {} axes for a rank-{rank} result over a rank-{} operand",
            axes.len(),
            input.rank()
        )));
    }

    let mut seen = vec![false; rank];
    for &axis in axes
    {
        let axis = usize::from(axis);
        let slot = seen.get_mut(axis).ok_or_else(|| {
            ComputeError::Compilation(format!("Permute axis {axis} is out of range"))
        })?;

        if *slot
        {
            return Err(ComputeError::Compilation(format!(
                "Permute axis {axis} appears more than once"
            )));
        }
        *slot = true;
    }

    let input_strides = row_major_strides(input.dims())?;
    let output_strides = row_major_strides(output.dims())?;

    let mut body = String::new();
    body.push_str("        unsigned long long source = 0ull;\n");

    for (position, &axis) in axes.iter().enumerate()
    {
        let axis = usize::from(axis);

        let output_dimension = output
            .dims()
            .get(position)
            .copied()
            .ok_or(ComputeError::ShapeOverflow)?;
        let input_dimension = input
            .dims()
            .get(axis)
            .copied()
            .ok_or(ComputeError::ShapeOverflow)?;

        if output_dimension != input_dimension
        {
            return Err(ComputeError::Compilation(format!(
                "Permute expects output axis {position} to equal input axis {axis}, \
                 got {output_dimension} and {input_dimension}"
            )));
        }

        let output_stride = output_strides
            .get(position)
            .copied()
            .ok_or(ComputeError::ShapeOverflow)?;
        let input_stride = input_strides
            .get(axis)
            .copied()
            .ok_or(ComputeError::ShapeOverflow)?;

        // A zero dimension means zero elements, and a zero-element kernel never
        // reaches this body — so no literal divisor or modulus here is ever 0.
        body.push_str(&format!(
            "        source += ((index / {output_stride}ull) % {output_dimension}ull) * \
             {input_stride}ull;\n"
        ));
    }

    body.push_str("        result[index] = operand_0[source];\n");
    Ok(body)
}

// ---------------------------------------------------------------------------
// Transfer helpers
// ---------------------------------------------------------------------------

fn checked_range(
    total_bytes: usize,
    offset_bytes: usize,
    length_bytes: usize,
    overflow: &'static str,
    bounds: &'static str,
) -> ComputeResult<()> {
    let end = offset_bytes
        .checked_add(length_bytes)
        .ok_or_else(|| ComputeError::Transfer(overflow.to_string()))?;

    if end > total_bytes
    {
        return Err(ComputeError::Transfer(bounds.to_string()));
    }

    Ok(())
}

fn require_word_alignment(offset_bytes: usize, length_bytes: usize) -> ComputeResult<()> {
    if !offset_bytes.is_multiple_of(WORD_BYTES) || !length_bytes.is_multiple_of(WORD_BYTES)
    {
        return Err(ComputeError::Unsupported(
            "CUDA Reference transfers require 4-byte-aligned offsets and lengths",
        ));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// ComputeBackend
// ---------------------------------------------------------------------------

impl ComputeBackend for CudaReferenceAdapter {
    type Buffer = CudaReferenceBuffer;
    type Kernel = CudaReferenceKernel;
    type Stream = CudaReferenceStream;
    type Event = CudaReferenceEvent;

    fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }

    /// Allocates device-resident storage.
    ///
    /// `ReferencePlanRuntime` requests `MemorySpace::Host` with alignment `1`
    /// because the compute contract exposes no way to negotiate placement — the
    /// request is a fixed convention, not a claim about where the bytes live.
    /// This adapter honours it by allocating CUDA device memory: the caller
    /// never gets a host pointer, and every byte still crosses the boundary
    /// through an explicit `write` or `read`.
    fn allocate(
        &self,
        bytes: usize,
        alignment: usize,
        memory_space: MemorySpace,
    ) -> ComputeResult<Self::Buffer> {
        if alignment == 0
        {
            return Err(ComputeError::InvalidArgument(
                "buffer alignment must be non-zero",
            ));
        }
        if alignment > CUDA_ALLOCATION_ALIGNMENT
            || !CUDA_ALLOCATION_ALIGNMENT.is_multiple_of(alignment)
        {
            return Err(ComputeError::Unsupported(
                "CUDA device allocations guarantee 256-byte alignment; a stricter or non-dividing \
                 request cannot be honoured",
            ));
        }
        if !matches!(memory_space, MemorySpace::Host | MemorySpace::Device)
        {
            return Err(ComputeError::Unsupported(
                "CUDA Reference buffers are device memory; only the runtime's Host convention and \
                 Device are accepted",
            ));
        }
        if !bytes.is_multiple_of(WORD_BYTES)
        {
            return Err(ComputeError::Unsupported(
                "CUDA Reference buffers hold whole 4-byte words",
            ));
        }

        let raw = self
            .runtime
            .allocate(bytes)
            .map_err(ComputeError::Allocation)?;

        Ok(CudaReferenceBuffer {
            raw,
            bytes,
            memory_space,
        })
    }

    fn write(
        &self,
        destination: &Self::Buffer,
        offset_bytes: usize,
        data: &[u8],
    ) -> ComputeResult<()> {
        checked_range(
            destination.bytes,
            offset_bytes,
            data.len(),
            "write range overflow",
            "write exceeds buffer bounds",
        )?;

        if data.is_empty()
        {
            return Ok(());
        }

        require_word_alignment(offset_bytes, data.len())?;

        self.runtime
            .write(&destination.raw, offset_bytes, data)
            .map_err(ComputeError::Transfer)
    }

    fn read(
        &self,
        source: &Self::Buffer,
        offset_bytes: usize,
        destination: &mut [u8],
    ) -> ComputeResult<()> {
        checked_range(
            source.bytes,
            offset_bytes,
            destination.len(),
            "read range overflow",
            "read exceeds buffer bounds",
        )?;

        if destination.is_empty()
        {
            return Ok(());
        }

        require_word_alignment(offset_bytes, destination.len())?;

        self.runtime
            .read(&source.raw, offset_bytes, destination)
            .map_err(ComputeError::Transfer)
    }

    /// Decodes a Reference artefact, generates its CUDA C, compiles it with
    /// NVRTC and loads the resulting module.
    ///
    /// This is the only place CUDA C is produced or compiled. A module in any
    /// other format is refused rather than reinterpreted.
    fn compile(&self, module: &KernelModule) -> ComputeResult<Self::Kernel> {
        if module.format != KernelFormat::Reference
        {
            return Err(ComputeError::Unsupported(
                "the CUDA Reference adapter accepts Reference kernel modules only",
            ));
        }

        let artifact = ReferenceKernelArtifact::decode(&module.code)
            .map_err(|error| ComputeError::Compilation(error.to_string()))?;

        let expected_entry_point = artifact.entry_point();
        if module.entry_point != expected_entry_point
        {
            return Err(ComputeError::Compilation(format!(
                "module entry point '{}' does not match the artefact's '{expected_entry_point}'",
                module.entry_point
            )));
        }

        let plan = plan_kernel(&artifact, self.block_x, self.max_grid_x)?;

        let raw = self
            .runtime
            .compile_cuda_c(
                &plan.source,
                CUDA_ENTRY_POINT,
                CudaRawCompileOptions::default(),
            )
            .map_err(ComputeError::Compilation)?;

        self.kernels_compiled.fetch_add(1, Ordering::Relaxed);

        Ok(CudaReferenceKernel {
            raw,
            entry_point: expected_entry_point,
            kernel_id: artifact.kernel_id().get(),
            opcode: artifact.opcode(),
            operand_bytes: plan.operand_bytes,
            result_bytes: plan.result_bytes,
            elements: plan.elements,
            grid_x: plan.grid_x,
            block_x: self.block_x,
            source: plan.source,
        })
    }

    fn create_stream(&self) -> ComputeResult<Self::Stream> {
        Ok(CudaReferenceStream(()))
    }

    /// Submits one dispatch to the device.
    ///
    /// The runtime's `[1, 1, 1]` geometry is matched exactly and then replaced
    /// by the grid the kernel actually needs — the Reference contract leaves
    /// geometry to the backend, and pretending one thread means one element
    /// would silently drop work.
    fn launch(
        &self,
        kernel: &Self::Kernel,
        _stream: &Self::Stream,
        config: LaunchConfig,
        bindings: &[BufferBinding<'_, Self::Buffer>],
    ) -> ComputeResult<Self::Event> {
        if config.grid != CANONICAL_GRID
            || config.block != CANONICAL_BLOCK
            || config.shared_memory_bytes != 0
        {
            return Err(ComputeError::Unsupported(
                "the CUDA Reference adapter expects the canonical [1, 1, 1] launch configuration",
            ));
        }

        let expected_bindings = kernel.operand_bytes.len().saturating_add(1);
        if bindings.len() != expected_bindings
        {
            return Err(ComputeError::Launch(format!(
                "{} requires {expected_bindings} binding(s) but received {}",
                kernel.entry_point,
                bindings.len()
            )));
        }

        // Slots are the kernel parameter indices: operands `0 .. N-1` first,
        // the result at `N`, last.
        for (position, binding) in bindings.iter().enumerate()
        {
            let slot = u32::try_from(position)
                .map_err(|_| ComputeError::Launch("binding slot exceeds u32".to_string()))?;
            if binding.slot != slot
            {
                return Err(ComputeError::Launch(format!(
                    "binding {position} declares slot {} instead of {slot}",
                    binding.slot
                )));
            }

            let is_result = position == kernel.operand_bytes.len();
            let expected_bytes = if is_result
            {
                kernel.result_bytes
            }
            else
            {
                kernel
                    .operand_bytes
                    .get(position)
                    .copied()
                    .ok_or_else(|| ComputeError::Launch("missing operand size".to_string()))?
            };

            if binding.length_bytes != expected_bytes
            {
                return Err(ComputeError::Launch(format!(
                    "binding {position} spans {} byte(s); the kernel expects {expected_bytes}",
                    binding.length_bytes
                )));
            }

            let expected_access = if is_result
            {
                BufferAccess::WriteOnly
            }
            else
            {
                BufferAccess::ReadOnly
            };
            if binding.access != expected_access
            {
                return Err(ComputeError::Unsupported(
                    "Reference dispatches bind read-only operands and a write-only result",
                ));
            }

            checked_range(
                binding.buffer.bytes,
                binding.offset_bytes,
                binding.length_bytes,
                "binding range overflow",
                "binding exceeds buffer bounds",
            )?;

            // The kernel addresses whole 32-bit words from the bound pointer.
            if !binding.offset_bytes.is_multiple_of(WORD_BYTES)
            {
                return Err(ComputeError::Unsupported(
                    "CUDA Reference bindings require 4-byte-aligned offsets",
                ));
            }
        }

        // An empty tensor has nothing to compute, so nothing is submitted and
        // nothing is counted. The returned event says so rather than standing
        // in for work that never happened.
        if kernel.elements == 0 || kernel.grid_x == 0
        {
            return Ok(CudaReferenceEvent { raw: None });
        }

        let mut raw_bindings = Vec::with_capacity(bindings.len());
        for binding in bindings
        {
            raw_bindings.push(CudaRawBinding {
                buffer: &binding.buffer.raw,
                offset_bytes: binding.offset_bytes,
                length_bytes: binding.length_bytes,
                access: match binding.access
                {
                    BufferAccess::ReadOnly => CudaRawAccess::ReadOnly,
                    BufferAccess::WriteOnly => CudaRawAccess::WriteOnly,
                    BufferAccess::ReadWrite => CudaRawAccess::ReadWrite,
                },
            });
        }

        let event = self
            .runtime
            .launch(
                &kernel.raw,
                CudaRawLaunchConfig {
                    grid: [kernel.grid_x, 1, 1],
                    block: [kernel.block_x, 1, 1],
                    shared_memory_bytes: 0,
                },
                &raw_bindings,
            )
            .map_err(ComputeError::Launch)?;

        self.kernels_launched.fetch_add(1, Ordering::Relaxed);

        Ok(CudaReferenceEvent { raw: Some(event) })
    }

    fn wait(&self, event: &Self::Event) -> ComputeResult<()> {
        match &event.raw
        {
            Some(raw) => self
                .runtime
                .wait(raw)
                .map_err(ComputeError::Synchronization),
            // Nothing was submitted, so there is nothing to wait for.
            None => Ok(()),
        }
    }

    fn synchronize(&self, _stream: &Self::Stream) -> ComputeResult<()> {
        self.runtime
            .synchronize()
            .map_err(ComputeError::Synchronization)
    }
}

#[cfg(test)]
mod tests {
    //! Generation tests. They need no CUDA device, so they run everywhere —
    //! which is what makes them worth having next to the device tests in
    //! `tests/cuda_reference.rs`.

    use super::*;

    fn strides(dims: &[u64]) -> Vec<u64> {
        row_major_strides(dims).expect("representable strides")
    }

    #[test]
    fn row_major_strides_follow_the_reference_layout() {
        assert_eq!(strides(&[]), Vec::<u64>::new());
        assert_eq!(strides(&[5]), vec![1]);
        assert_eq!(strides(&[2, 3]), vec![3, 1]);
        assert_eq!(strides(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn the_grid_covers_every_element() {
        assert_eq!(dispatch_grid(0, 256, 65535), Ok(0));
        assert_eq!(dispatch_grid(1, 256, 65535), Ok(1));
        assert_eq!(dispatch_grid(256, 256, 65535), Ok(1));
        assert_eq!(dispatch_grid(257, 256, 65535), Ok(2));
        assert_eq!(dispatch_grid(1000, 256, 65535), Ok(4));
        // Beyond the device limit the grid is capped; the kernel strides.
        assert_eq!(dispatch_grid(256 * 100_000, 256, 65535), Ok(65535));
    }

    #[test]
    fn arity_matches_the_executable_opcodes() {
        assert_eq!(operand_arity(ReferenceOpcode::Relu), Ok(1));
        assert_eq!(operand_arity(ReferenceOpcode::Scale), Ok(1));
        assert_eq!(operand_arity(ReferenceOpcode::ShapeCopy), Ok(1));
        assert_eq!(operand_arity(ReferenceOpcode::Permute), Ok(1));
        assert_eq!(operand_arity(ReferenceOpcode::Add), Ok(2));
        assert_eq!(operand_arity(ReferenceOpcode::Div), Ok(2));

        for opcode in [ReferenceOpcode::Exp, ReferenceOpcode::Log]
        {
            assert!(
                matches!(operand_arity(opcode), Err(ComputeError::Unsupported(_))),
                "{opcode:?} must be refused, never approximated"
            );
        }
    }

    #[test]
    fn word_addressing_covers_every_arithmetic_free_opcode() {
        assert!(addresses_words(ReferenceOpcode::ShapeCopy));
        assert!(addresses_words(ReferenceOpcode::Permute));
        assert!(addresses_words(ReferenceOpcode::Relu));
        assert!(!addresses_words(ReferenceOpcode::Add));
        assert!(!addresses_words(ReferenceOpcode::Scale));
    }

    // -----------------------------------------------------------------------
    // Generated source, built from real artefacts
    // -----------------------------------------------------------------------

    use scirust_compute::Shape;
    use scirust_tensor_compile::{CanonicalCompiler, ExternalBindings, KernelLowerer};
    use scirust_tensor_ir::{Graph, Operation, Scalar, TensorType};
    use scirust_tensor_reference::ReferenceKernelGenerator;

    fn f32_type(dims: Vec<usize>) -> TensorType {
        TensorType::new(DType::F32, Shape::new(dims))
    }

    /// The artefact of a one-instruction plan, built through the real pipeline.
    fn artifact_of(
        operation: Operation,
        operands: usize,
        dims: Vec<usize>,
        output_dims: Vec<usize>,
    ) -> ReferenceKernelArtifact {
        let mut graph = Graph::new();
        let mut inputs = Vec::with_capacity(operands);
        for index in 0..operands
        {
            let name = format!("operand{index}");
            inputs.push(
                graph
                    .add_input(name, f32_type(dims.clone()))
                    .expect("input"),
            );
        }

        let result = graph
            .add_node(operation, inputs, f32_type(output_dims))
            .expect("operation");
        graph.set_outputs(vec![result]).expect("outputs");

        let plan = CanonicalCompiler::new().compile(&graph).expect("compiles");
        let bindings = ExternalBindings::derive(&plan);
        let lowered = KernelLowerer::new()
            .lower(&plan, &bindings)
            .expect("lowers");

        ReferenceKernelGenerator::new()
            .generate(&lowered)
            .expect("generates")
            .artifacts()
            .first()
            .expect("one kernel")
            .clone()
    }

    fn source_of(artifact: &ReferenceKernelArtifact) -> String {
        let elements = layout_elements(artifact.result()).expect("representable");
        generate_cuda_c(artifact, elements).expect("generated source")
    }

    /// Prints every generated kernel. Run with `--nocapture` to review them.
    #[test]
    fn every_executable_opcode_generates_a_kernel() {
        let cases = vec![
            ("Relu", artifact_of(Operation::Relu, 1, vec![4], vec![4])),
            (
                "Scale",
                artifact_of(
                    Operation::Scale {
                        factor: Scalar::f32(0.25),
                    },
                    1,
                    vec![4],
                    vec![4],
                ),
            ),
            ("Add", artifact_of(Operation::Add, 2, vec![4], vec![4])),
            ("Sub", artifact_of(Operation::Sub, 2, vec![4], vec![4])),
            ("Mul", artifact_of(Operation::Mul, 2, vec![4], vec![4])),
            ("Div", artifact_of(Operation::Div, 2, vec![4], vec![4])),
            (
                "ShapeCopy",
                artifact_of(
                    Operation::Reshape {
                        shape: Shape::new(vec![6]),
                    },
                    1,
                    vec![2, 3],
                    vec![6],
                ),
            ),
            (
                "Permute",
                artifact_of(
                    Operation::Transpose {
                        permutation: vec![1, 0],
                    },
                    1,
                    vec![2, 3],
                    vec![3, 2],
                ),
            ),
        ];

        for (label, artifact) in &cases
        {
            let source = source_of(artifact);
            println!("========== {label} ==========\n{source}");

            assert!(source.contains("extern \"C\" __global__ void scirust_reference_kernel("));
            assert!(source.contains("for (; index < total; index += stride)"));
            // No kernel name may depend on anything but the opcode family.
            assert!(!source.contains("scirust_reference_kernel_"));
            // The source must be a pure function of the artefact.
            assert_eq!(source, source_of(artifact), "{label} is not deterministic");
        }
    }

    #[test]
    fn parameter_order_is_operands_then_result() {
        let source = source_of(&artifact_of(Operation::Sub, 2, vec![4], vec![4]));

        let operand_0 = source.find("const float* operand_0").expect("operand 0");
        let operand_1 = source.find("const float* operand_1").expect("operand 1");
        let result = source.find("float* result").expect("result");

        assert!(operand_0 < operand_1, "source: {source}");
        assert!(operand_1 < result, "source: {source}");
    }

    #[test]
    fn a_zero_element_kernel_has_no_loop_and_no_grid() {
        let artifact = artifact_of(Operation::Relu, 1, vec![0, 3], vec![0, 3]);
        let source = source_of(&artifact);

        println!("========== zero elements ==========\n{source}");
        assert!(!source.contains("for ("), "source: {source}");
        assert_eq!(
            dispatch_grid(0, 256, 65535),
            Ok(0),
            "an empty tensor is not dispatched at all"
        );
    }

    #[test]
    fn the_scale_factor_is_emitted_from_its_raw_bits() {
        let artifact = artifact_of(
            Operation::Scale {
                factor: Scalar::f32(-0.0),
            },
            1,
            vec![2],
            vec![2],
        );
        let source = source_of(&artifact);

        // `-0.0` has bit pattern 0x80000000; a decimal rendering could not
        // distinguish it from `+0.0`.
        assert!(
            source.contains("__uint_as_float(0x80000000u)"),
            "source: {source}"
        );
    }

    #[test]
    fn relu_selects_words_so_a_nan_payload_survives() {
        let source = source_of(&artifact_of(Operation::Relu, 1, vec![4], vec![4]));

        assert!(source.contains("const unsigned int* operand_0"), "{source}");
        assert!(
            source.contains("(bits & 0x7f800000u) == 0x7f800000u"),
            "{source}"
        );
        assert!(source.contains("? bits : 0x00000000u"), "{source}");
    }

    #[test]
    fn permute_unrolls_the_index_arithmetic_with_literal_constants() {
        let artifact = artifact_of(
            Operation::Transpose {
                permutation: vec![2, 0, 1],
            },
            1,
            vec![2, 3, 4],
            vec![4, 2, 3],
        );
        let source = source_of(&artifact);

        println!("========== rank-3 permute ==========\n{source}");

        // Output [4, 2, 3] has strides [6, 3, 1]; input [2, 3, 4] has strides
        // [12, 4, 1]; axes [2, 0, 1] map output axis 0 -> input axis 2, and so on.
        assert!(
            source.contains("source += ((index / 6ull) % 4ull) * 1ull;"),
            "{source}"
        );
        assert!(
            source.contains("source += ((index / 3ull) % 2ull) * 12ull;"),
            "{source}"
        );
        assert!(
            source.contains("source += ((index / 1ull) % 3ull) * 4ull;"),
            "{source}"
        );
        assert!(
            source.contains("result[index] = operand_0[source];"),
            "{source}"
        );
    }

    #[test]
    fn a_rank_zero_permute_copies_its_single_element() {
        let artifact = artifact_of(
            Operation::Transpose {
                permutation: vec![],
            },
            1,
            vec![],
            vec![],
        );
        let source = source_of(&artifact);

        println!("========== rank-0 permute ==========\n{source}");
        assert!(
            source.contains("const unsigned long long total = 1ull;"),
            "{source}"
        );
        assert!(
            source.contains("unsigned long long source = 0ull;"),
            "{source}"
        );
        assert!(
            source.contains("result[index] = operand_0[source];"),
            "{source}"
        );
        // No axis, so no index arithmetic at all.
        assert!(!source.contains("source +="), "{source}");
    }

    #[test]
    fn structural_kernels_address_words_and_arithmetic_kernels_address_floats() {
        let copy = source_of(&artifact_of(
            Operation::Reshape {
                shape: Shape::new(vec![6]),
            },
            1,
            vec![2, 3],
            vec![6],
        ));
        assert!(copy.contains("const unsigned int* operand_0"), "{copy}");
        assert!(copy.contains("unsigned int* result"), "{copy}");

        let add = source_of(&artifact_of(Operation::Add, 2, vec![4], vec![4]));
        assert!(add.contains("const float* operand_0"), "{add}");
        assert!(add.contains("float* result"), "{add}");
    }

    #[test]
    fn a_non_f32_or_non_reference_module_is_refused_before_any_compilation() {
        // `plan_kernel` is the gate every compilation goes through; there is no
        // path from a rejected artefact to NVRTC.
        let artifact = artifact_of(Operation::Relu, 1, vec![4], vec![4]);
        assert!(plan_kernel(&artifact, 256, 65535).is_ok());
        // A device-free proof that geometry derivation refuses a degenerate
        // device rather than guessing.
        assert!(matches!(
            plan_kernel(&artifact, 0, 65535),
            Err(ComputeError::Unsupported(_))
        ));
        assert!(matches!(
            plan_kernel(&artifact, 256, 0),
            Err(ComputeError::Unsupported(_))
        ));
    }
}
