//! Executes canonical Reference kernels on a WGPU device.
//!
//! # Why a second adapter
//!
//! [`crate::WgpuComputeAdapter`] speaks WGSL: it takes a
//! [`KernelFormat::Wgsl`] module and runs it. That is a different contract from
//! the one the canonical tensor pipeline emits, which is
//! [`KernelFormat::Reference`] plus the placement conventions
//! `scirust_tensor_runtime::ReferencePlanRuntime` fixes. Five of those
//! conventions the WGSL adapter deliberately rejects — `MemorySpace::Host`,
//! alignment `1`, `BufferAccess::WriteOnly`, zero-length bindings, and the
//! canonical `[1, 1, 1]` launch geometry. Bending it to accept them would
//! weaken a contract other callers rely on, so this module implements the
//! Reference contract separately and leaves that adapter untouched.
//!
//! # What it does
//!
//! ```text
//! KernelModule::Reference
//!   -> ReferenceKernelArtifact::decode
//!   -> validation
//!   -> deterministic, specialised WGSL
//!   -> shader module + compute pipeline      (once, during compile)
//!   -> bind group + dispatch + submit        (per launch)
//!   -> staging copy + map + read back        (per read)
//! ```
//!
//! One shader per logical kernel. The lowerer already deduplicates kernels by
//! full signature, so the element count, the shape, the strides, the `Scale`
//! factor and the permutation are all compile-time constants — no uniform
//! buffer, no push constants, no bytecode interpreter in the shader.
//!
//! **No WGSL is generated during execution**, and nothing here ever falls back
//! to the CPU: an opcode this adapter cannot run is a typed error, never a
//! quietly host-computed result.
//!
//! # Determinism
//!
//! The generated source is a pure function of the artefact: same artefact, same
//! bytes. Binding order is argument order. The `Scale` factor is emitted as
//! `bitcast<f32>(0x…u)` from its raw bits, never as a formatted decimal. There
//! is no clock, no random name, no address and no hash map anywhere in the
//! generation path.
//!
//! Execution determinism is narrower, and stated honestly:
//!
//! * `ShapeCopy` and `Permute` move `u32` words and never touch the float unit,
//!   so their results are **bit-identical** to the CPU interpreter's — NaN
//!   payloads, signed zeros, infinities and subnormals included.
//! * `Relu`, `Scale`, `Add`, `Sub`, `Mul` and `Div` are `f32` arithmetic. WGSL
//!   does not forbid contraction and drivers differ, so bit-identical results
//!   across CPU, a hardware GPU and a software adapter are **not promised**. A
//!   NaN operand yields a NaN result, but its payload may be canonicalised.

use core::fmt;
use std::{borrow::Cow, num::NonZeroU64, sync::mpsc};

use scirust_compute::{
    BufferAccess, BufferBinding, ComputeBackend, ComputeError, ComputeResult, DType,
    DeviceCapabilities, DeviceId, DeviceKind, KernelFormat, KernelModule, LaunchConfig,
    MemorySpace,
};
use scirust_tensor_reference::{
    ReferenceAttributes, ReferenceKernelArtifact, ReferenceOpcode, ReferenceTensorLayout,
};

/// Invocations per workgroup. Well under `Limits::downlevel_defaults()`'s
/// `max_compute_invocations_per_workgroup` of 256, so every conforming device
/// accepts it.
const WORKGROUP_SIZE: u32 = 64;

/// Storage width of one `f32`, and of the `u32` word the structural kernels
/// move instead.
const WORD_BYTES: usize = 4;

/// The launch geometry `ReferencePlanRuntime` issues.
///
/// It is a placeholder, not a description of the work: the Reference contract
/// leaves geometry to the backend, exactly as the CPU adapter derives its loop
/// bounds from the artefact rather than from this. Accepting anything else
/// would mean guessing at a caller's intent, so it is matched exactly and the
/// real grid comes from the kernel's own element count.
const CANONICAL_GRID: [u32; 3] = [1, 1, 1];
const CANONICAL_BLOCK: [u32; 3] = [1, 1, 1];

// ---------------------------------------------------------------------------
// Adapter construction
// ---------------------------------------------------------------------------

/// Which adapter to ask WGPU for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum WgpuPowerPreference {
    /// Let WGPU choose.
    #[default]
    Default,
    LowPower,
    HighPerformance,
}

/// How [`WgpuReferenceAdapter::request`] acquires its device.
///
/// Nothing here is implicit: a caller that wants a software adapter has to say
/// so, and a caller that does not will never silently get one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct WgpuReferenceOptions {
    pub power_preference: WgpuPowerPreference,
    /// Accept WGPU's fallback (software) adapter. `false` by default.
    ///
    /// Note that this only controls whether WGPU is *asked* for a fallback: a
    /// system whose only driver is a software one (Mesa lavapipe, for instance)
    /// reports it as its ordinary adapter either way. Read
    /// [`WgpuReferenceAdapter::adapter_info`] to learn what was actually
    /// acquired.
    pub allow_fallback_adapter: bool,
}

/// What kind of device WGPU actually handed over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WgpuDeviceClass {
    DiscreteGpu,
    IntegratedGpu,
    VirtualGpu,
    /// A software rasteriser such as Mesa lavapipe. Still a genuine WGPU
    /// device — the work runs through the driver, not through this crate's CPU
    /// interpreter.
    SoftwareCpu,
    Other,
}

impl WgpuDeviceClass {
    /// Whether the device is backed by real graphics hardware.
    pub const fn is_hardware(self) -> bool {
        matches!(
            self,
            Self::DiscreteGpu | Self::IntegratedGpu | Self::VirtualGpu
        )
    }
}

/// Honest identity of the acquired adapter.
///
/// Every field comes from `wgpu::AdapterInfo`; nothing is inferred or invented.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WgpuReferenceAdapterInfo {
    pub name: String,
    pub backend: String,
    pub driver: String,
    pub class: WgpuDeviceClass,
}

// ---------------------------------------------------------------------------
// Adapter
// ---------------------------------------------------------------------------

/// A WGPU device that executes canonical Reference kernels.
///
/// See the module documentation for the contract, the determinism guarantees
/// and their limits.
pub struct WgpuReferenceAdapter {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: WgpuReferenceAdapterInfo,
    capabilities: DeviceCapabilities,
    binding_alignment: usize,
    max_workgroups_per_dimension: u32,
}

impl WgpuReferenceAdapter {
    /// Acquire a device with WGPU's default preferences and no fallback.
    ///
    /// Returns [`ComputeError::BackendUnavailable`] when no adapter exists —
    /// never a silent substitute.
    pub fn new() -> ComputeResult<Self> {
        Self::request(WgpuReferenceOptions::default())
    }

    /// Acquire a device with explicit preferences.
    pub fn request(options: WgpuReferenceOptions) -> ComputeResult<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let power_preference = match options.power_preference
        {
            WgpuPowerPreference::Default => wgpu::PowerPreference::None,
            WgpuPowerPreference::LowPower => wgpu::PowerPreference::LowPower,
            WgpuPowerPreference::HighPerformance => wgpu::PowerPreference::HighPerformance,
        };

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference,
            force_fallback_adapter: options.allow_fallback_adapter,
            compatible_surface: None,
        }))
        .ok_or(ComputeError::BackendUnavailable(DeviceKind::Wgpu))?;

        let raw_info = adapter.get_info();

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("scirust-wgpu-reference"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))
        .map_err(|error| ComputeError::Allocation(error.to_string()))?;

        let info = WgpuReferenceAdapterInfo {
            name: raw_info.name.clone(),
            backend: format!("{:?}", raw_info.backend),
            driver: raw_info.driver.clone(),
            class: match raw_info.device_type
            {
                wgpu::DeviceType::DiscreteGpu => WgpuDeviceClass::DiscreteGpu,
                wgpu::DeviceType::IntegratedGpu => WgpuDeviceClass::IntegratedGpu,
                wgpu::DeviceType::VirtualGpu => WgpuDeviceClass::VirtualGpu,
                wgpu::DeviceType::Cpu => WgpuDeviceClass::SoftwareCpu,
                wgpu::DeviceType::Other => WgpuDeviceClass::Other,
            },
        };

        let limits = device.limits();

        let capabilities = DeviceCapabilities {
            device: DeviceId::new(DeviceKind::Wgpu, 0),
            name: format!("scirust-wgpu-reference: {} ({})", info.name, info.backend),
            supported_dtypes: vec![DType::F32],
            max_buffer_bytes: usize::try_from(limits.max_storage_buffer_binding_size).ok(),
            max_workgroup_size: [
                limits.max_compute_workgroup_size_x,
                limits.max_compute_workgroup_size_y,
                limits.max_compute_workgroup_size_z,
            ],
            supports_async_execution: true,
        };

        Ok(Self {
            device,
            queue,
            info,
            capabilities,
            binding_alignment: limits.min_storage_buffer_offset_alignment as usize,
            max_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        })
    }

    /// What WGPU actually handed over: name, backend, driver, hardware class.
    pub const fn adapter_info(&self) -> &WgpuReferenceAdapterInfo {
        &self.info
    }

    /// Capabilities reported by the acquired device.
    pub const fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }

    /// Drain any pending validation error scope into a typed failure.
    fn take_error(&self, wrap: fn(String) -> ComputeError) -> ComputeResult<()> {
        match pollster::block_on(self.device.pop_error_scope())
        {
            Some(error) => Err(wrap(error.to_string())),
            None => Ok(()),
        }
    }
}

impl fmt::Debug for WgpuReferenceAdapter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WgpuReferenceAdapter")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Associated types
// ---------------------------------------------------------------------------

/// Device-resident buffer owned by the Reference adapter.
///
/// `bytes` is the *logical* size the runtime asked for. WGPU cannot allocate
/// zero bytes and rounds nothing else here, so a physical buffer may be larger;
/// every read, write and binding stays bounded by the logical size, and the
/// extra bytes are unreachable through this API.
pub struct WgpuReferenceBuffer {
    raw: wgpu::Buffer,
    bytes: usize,
    memory_space: MemorySpace,
}

impl WgpuReferenceBuffer {
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

impl fmt::Debug for WgpuReferenceBuffer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WgpuReferenceBuffer")
            .field("bytes", &self.bytes)
            .field("memory_space", &self.memory_space)
            .finish_non_exhaustive()
    }
}

/// One Reference kernel, compiled into a WGPU pipeline.
///
/// Everything invariant lives here: the pipeline, the binding ABI, the logical
/// byte sizes and the dispatch geometry. Execution adds nothing to it.
pub struct WgpuReferenceKernel {
    pipeline: wgpu::ComputePipeline,
    entry_point: String,
    kernel_id: u32,
    opcode: ReferenceOpcode,
    operand_bytes: Vec<usize>,
    result_bytes: usize,
    elements: usize,
    workgroups: u32,
    wgsl: String,
}

impl WgpuReferenceKernel {
    pub fn entry_point(&self) -> &str {
        &self.entry_point
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

    /// Workgroups dispatched per launch. Zero for an empty tensor, which is
    /// dispatched not at all rather than dispatched emptily.
    pub const fn workgroups(&self) -> u32 {
        self.workgroups
    }

    /// The generated source, for diagnostics and determinism tests.
    pub fn wgsl(&self) -> &str {
        &self.wgsl
    }
}

impl fmt::Debug for WgpuReferenceKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WgpuReferenceKernel")
            .field("entry_point", &self.entry_point)
            .field("opcode", &self.opcode)
            .field("elements", &self.elements)
            .field("workgroups", &self.workgroups)
            .finish_non_exhaustive()
    }
}

/// Logical stream mapped onto the adapter's single WGPU queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgpuReferenceStream(());

/// A real queue submission.
#[derive(Debug, Clone)]
pub struct WgpuReferenceEvent {
    submission: wgpu::SubmissionIndex,
}

// ---------------------------------------------------------------------------
// WGSL generation
// ---------------------------------------------------------------------------

/// Everything the generator needs, extracted and checked once.
struct KernelPlan {
    operand_bytes: Vec<usize>,
    result_bytes: usize,
    elements: usize,
    workgroups: u32,
    wgsl: String,
}

/// Element count of a layout, as a `usize` that also fits `u32`.
///
/// The shader indexes with `u32`, so a tensor beyond that is rejected rather
/// than silently truncated.
fn layout_elements(layout: &ReferenceTensorLayout) -> ComputeResult<usize> {
    let elements = layout.elements();

    if elements > u64::from(u32::MAX)
    {
        return Err(ComputeError::Unsupported(
            "WGPU Reference kernels index with u32; this tensor has too many elements",
        ));
    }

    usize::try_from(elements).map_err(|_| ComputeError::ShapeOverflow)
}

fn byte_size(elements: usize) -> ComputeResult<usize> {
    elements
        .checked_mul(WORD_BYTES)
        .ok_or(ComputeError::ByteSizeOverflow)
}

/// Row-major strides of `dims`, each fitting `u32`.
fn row_major_strides(dims: &[u64]) -> ComputeResult<Vec<u32>> {
    let rank = dims.len();
    let mut strides = vec![1u32; rank];

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
            .and_then(|value| u32::try_from(value).ok())
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
            "this Reference opcode has no WGPU kernel",
        )),
    }
}

/// Whether the opcode moves raw words rather than computing on floats.
const fn is_structural(opcode: ReferenceOpcode) -> bool {
    matches!(
        opcode,
        ReferenceOpcode::ShapeCopy | ReferenceOpcode::Permute
    )
}

/// A WGSL identifier this generator is willing to emit.
fn valid_identifier(name: &str) -> bool {
    let mut characters = name.chars();

    match characters.next()
    {
        Some(first) if first.is_ascii_alphabetic() || first == '_' =>
        {
            characters.all(|character| character.is_ascii_alphanumeric() || character == '_')
        },
        _ => false,
    }
}

/// Validates one artefact and generates its specialised shader.
fn plan_kernel(
    artifact: &ReferenceKernelArtifact,
    entry_point: &str,
    max_workgroups: u32,
) -> ComputeResult<KernelPlan> {
    if artifact.dtype() != DType::F32
    {
        return Err(ComputeError::Unsupported(
            "WGPU Reference kernels execute F32 only",
        ));
    }
    if !valid_identifier(entry_point)
    {
        return Err(ComputeError::Compilation(format!(
            "entry point '{entry_point}' is not a WGSL identifier"
        )));
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

    let workgroups = dispatch_workgroups(elements, max_workgroups)?;
    let wgsl = generate_wgsl(artifact, entry_point, elements, workgroups)?;

    Ok(KernelPlan {
        operand_bytes,
        result_bytes,
        elements,
        workgroups,
        wgsl,
    })
}

/// Workgroups needed to cover `elements`, capped by the device limit.
///
/// The cap is not a truncation: the shader strides by the total invocation
/// count, so a capped grid still covers every element.
fn dispatch_workgroups(elements: usize, max_workgroups: u32) -> ComputeResult<u32> {
    if elements == 0
    {
        return Ok(0);
    }
    if max_workgroups == 0
    {
        return Err(ComputeError::Unsupported(
            "the device advertises no compute workgroup per dimension",
        ));
    }

    let size = WORKGROUP_SIZE as usize;
    let needed = elements.div_ceil(size);
    let needed = u32::try_from(needed).unwrap_or(u32::MAX);

    Ok(needed.min(max_workgroups))
}

/// The shader source for one artefact. A pure function of its inputs.
fn generate_wgsl(
    artifact: &ReferenceKernelArtifact,
    entry_point: &str,
    elements: usize,
    workgroups: u32,
) -> ComputeResult<String> {
    let opcode = artifact.opcode();
    let word = if is_structural(opcode) { "u32" } else { "f32" };

    let mut source = String::new();
    source.push_str("// Generated by scirust-gpu from a Reference kernel artefact.\n");
    source.push_str(&format!("// opcode: {opcode:?}\n"));
    source.push_str(&format!("// elements: {elements}\n\n"));

    for (index, _) in artifact.operands().iter().enumerate()
    {
        source.push_str(&format!(
            "@group(0) @binding({index}) var<storage, read> operand_{index} : array<{word}>;\n"
        ));
    }
    source.push_str(&format!(
        "@group(0) @binding({}) var<storage, read_write> result : array<{word}>;\n\n",
        artifact.operands().len()
    ));

    source.push_str(&format!("@compute @workgroup_size({WORKGROUP_SIZE})\n"));
    source.push_str(&format!(
        "fn {entry_point}(@builtin(global_invocation_id) global_id : vec3<u32>) {{\n"
    ));

    if elements == 0
    {
        // Nothing to compute. The pipeline exists so `compile` stays uniform;
        // `launch` issues no dispatch for it at all.
        source.push_str("}\n");
        return Ok(source);
    }

    // A capped grid still covers every element: each invocation strides by the
    // total invocation count.
    let stride = u64::from(workgroups.max(1)) * u64::from(WORKGROUP_SIZE);
    let stride = u32::try_from(stride).unwrap_or(u32::MAX);

    source.push_str("    var index : u32 = global_id.x;\n");
    source.push_str("    loop {\n");
    source.push_str(&format!("        if (index >= {elements}u) {{ break; }}\n"));
    source.push_str(&kernel_body(artifact)?);
    source.push_str(&format!("        index = index + {stride}u;\n"));
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
            // interpreter's exact rule. The NaN test is bitwise so no driver
            // can optimise it away under a finite-math assumption.
            body.push_str("        let value : f32 = operand_0[index];\n");
            body.push_str("        let bits : u32 = bitcast<u32>(value);\n");
            body.push_str(
                "        let is_nan : bool = ((bits & 0x7f800000u) == 0x7f800000u) && \
                 ((bits & 0x007fffffu) != 0u);\n",
            );
            body.push_str("        if (is_nan || value > 0.0) {\n");
            body.push_str("            result[index] = value;\n");
            body.push_str("        } else {\n");
            body.push_str("            result[index] = 0.0;\n");
            body.push_str("        }\n");
        },
        ReferenceOpcode::Scale =>
        {
            let factor_bits = match artifact.attributes()
            {
                ReferenceAttributes::Scale { factor_bits } => *factor_bits,
                _ =>
                {
                    return Err(ComputeError::Compilation(
                        "Scale artefact carries no factor".to_string(),
                    ));
                },
            };

            // Emitted from raw bits, never as a formatted decimal, so the
            // constant is exactly the one the artefact stores.
            body.push_str(&format!(
                "        result[index] = operand_0[index] * bitcast<f32>({factor_bits:#010x}u);\n"
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
                        "this Reference opcode has no WGPU kernel",
                    ));
                },
            };
            body.push_str(&format!(
                "        result[index] = operand_0[index] {operator} operand_1[index];\n"
            ));
        },
        ReferenceOpcode::ShapeCopy =>
        {
            // `u32` words: the float unit is never involved, so every bit
            // survives — NaN payloads, signed zeros, infinities, subnormals.
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
                "this Reference opcode has no WGPU kernel",
            ));
        },
    }

    Ok(body)
}

/// Index arithmetic for `Permute`, fully unrolled with literal constants.
///
/// Convention, the same one the lowerer validates:
/// `output.shape[i] == input.shape[axes[i]]`. For an output element at linear
/// index `o`, the source index is
/// `sum_i ((o / out_stride[i]) % out_dim[i]) * in_stride[axes[i]]`.
fn permute_body(artifact: &ReferenceKernelArtifact) -> ComputeResult<String> {
    let axes = match artifact.attributes()
    {
        ReferenceAttributes::Permute { axes } => axes,
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
    body.push_str("        var source : u32 = 0u;\n");

    for (position, &axis) in axes.iter().enumerate()
    {
        let axis = usize::from(axis);

        let output_dimension = output
            .dims()
            .get(position)
            .copied()
            .and_then(|value| u32::try_from(value).ok())
            .ok_or(ComputeError::ShapeOverflow)?;
        let input_dimension = input
            .dims()
            .get(axis)
            .copied()
            .and_then(|value| u32::try_from(value).ok())
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
            "        source = source + (((index / {output_stride}u) % {output_dimension}u) * \
             {input_stride}u);\n"
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
            "WGPU transfers require 4-byte-aligned offsets and lengths",
        ));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// ComputeBackend
// ---------------------------------------------------------------------------

impl ComputeBackend for WgpuReferenceAdapter {
    type Buffer = WgpuReferenceBuffer;
    type Kernel = WgpuReferenceKernel;
    type Stream = WgpuReferenceStream;
    type Event = WgpuReferenceEvent;

    fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }

    /// Allocates device-visible storage.
    ///
    /// `ReferencePlanRuntime` requests `MemorySpace::Host` with alignment `1`
    /// because the compute contract exposes no way to negotiate placement — the
    /// request is a fixed convention, not a claim about where the bytes live.
    /// This adapter honours it by allocating a WGPU storage buffer: the caller
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
        if alignment > WORD_BYTES || !WORD_BYTES.is_multiple_of(alignment)
        {
            return Err(ComputeError::Unsupported(
                "WGPU buffers guarantee 4-byte alignment; a stricter request cannot be honoured",
            ));
        }
        if !matches!(memory_space, MemorySpace::Host | MemorySpace::Device)
        {
            return Err(ComputeError::Unsupported(
                "WGPU Reference buffers are device storage; only the runtime's Host convention \
                 and Device are accepted",
            ));
        }
        if !bytes.is_multiple_of(WORD_BYTES)
        {
            return Err(ComputeError::Unsupported(
                "WGPU Reference buffers hold whole 4-byte words",
            ));
        }
        if let Some(maximum) = self.capabilities.max_buffer_bytes
        {
            if bytes > maximum
            {
                return Err(ComputeError::Allocation(format!(
                    "requested {bytes} bytes exceeds device limit {maximum}"
                )));
            }
        }

        // WGPU cannot allocate zero bytes. The logical size stays `bytes`, so
        // the padding is unreachable: nothing may read or write past it.
        let physical = bytes.max(WORD_BYTES);
        let size =
            u64::try_from(physical).map_err(|error| ComputeError::Allocation(error.to_string()))?;

        self.device.push_error_scope(wgpu::ErrorFilter::Validation);

        let raw = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scirust-wgpu-reference-buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.take_error(ComputeError::Allocation)?;

        Ok(WgpuReferenceBuffer {
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

        let offset = u64::try_from(offset_bytes)
            .map_err(|error| ComputeError::Transfer(error.to_string()))?;

        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        self.queue.write_buffer(&destination.raw, offset, data);
        self.take_error(ComputeError::Transfer)
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

        let offset = u64::try_from(offset_bytes)
            .map_err(|error| ComputeError::Transfer(error.to_string()))?;
        let length = u64::try_from(destination.len())
            .map_err(|error| ComputeError::Transfer(error.to_string()))?;

        self.device.push_error_scope(wgpu::ErrorFilter::Validation);

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scirust-wgpu-reference-readback"),
            size: length,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scirust-wgpu-reference-read"),
            });
        encoder.copy_buffer_to_buffer(&source.raw, offset, &staging, 0, length);
        let submission = self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        let _ = self
            .device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(submission));

        receiver
            .recv()
            .map_err(|error| ComputeError::Transfer(error.to_string()))?
            .map_err(|error| ComputeError::Transfer(error.to_string()))?;

        self.take_error(ComputeError::Transfer)?;

        let mapped = slice.get_mapped_range();
        if mapped.len() != destination.len()
        {
            // Defensive: the staging buffer was created at exactly this length.
            drop(mapped);
            staging.unmap();
            return Err(ComputeError::Transfer(
                "mapped readback range does not match the requested length".to_string(),
            ));
        }
        destination.copy_from_slice(&mapped);
        drop(mapped);
        staging.unmap();

        Ok(())
    }

    /// Decodes a Reference artefact, generates its shader and builds a pipeline.
    ///
    /// This is the only place WGSL is produced. A module in any other format is
    /// refused rather than reinterpreted.
    fn compile(&self, module: &KernelModule) -> ComputeResult<Self::Kernel> {
        if module.format != KernelFormat::Reference
        {
            return Err(ComputeError::Unsupported(
                "the WGPU Reference adapter accepts Reference kernel modules only",
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

        let plan = plan_kernel(
            &artifact,
            &expected_entry_point,
            self.max_workgroups_per_dimension,
        )?;

        self.device.push_error_scope(wgpu::ErrorFilter::Validation);

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(expected_entry_point.as_str()),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(plan.wgsl.as_str())),
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(expected_entry_point.as_str()),
                layout: None,
                module: &shader,
                entry_point: expected_entry_point.as_str(),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        self.take_error(ComputeError::Compilation)?;

        Ok(WgpuReferenceKernel {
            pipeline,
            entry_point: expected_entry_point,
            kernel_id: artifact.kernel_id().get(),
            opcode: artifact.opcode(),
            operand_bytes: plan.operand_bytes,
            result_bytes: plan.result_bytes,
            elements: plan.elements,
            workgroups: plan.workgroups,
            wgsl: plan.wgsl,
        })
    }

    fn create_stream(&self) -> ComputeResult<Self::Stream> {
        Ok(WgpuReferenceStream(()))
    }

    /// Encodes and submits one dispatch.
    ///
    /// The runtime's `[1, 1, 1]` geometry is matched exactly and then replaced
    /// by the grid the kernel actually needs — the Reference contract leaves
    /// geometry to the backend, and pretending one workgroup means one element
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
                "the WGPU Reference adapter expects the canonical [1, 1, 1] launch configuration",
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

        // Slots are the argument indices: operands first, the result last.
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

            // WGSL has no write-only storage: the result is declared
            // `read_write` and only ever written. Read operands must stay read.
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

            if !binding.offset_bytes.is_multiple_of(self.binding_alignment)
            {
                return Err(ComputeError::Unsupported(
                    "WGPU storage-buffer offsets must satisfy the device alignment",
                ));
            }
        }

        self.device.push_error_scope(wgpu::ErrorFilter::Validation);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(kernel.entry_point.as_str()),
            });

        // An empty tensor has nothing to compute. The submission still happens,
        // so the returned event is a real one that `wait` can honour.
        if kernel.workgroups > 0
        {
            let mut entries = Vec::with_capacity(bindings.len());
            for binding in bindings
            {
                let offset = u64::try_from(binding.offset_bytes)
                    .map_err(|error| ComputeError::Launch(error.to_string()))?;
                let length = u64::try_from(binding.length_bytes)
                    .map_err(|error| ComputeError::Launch(error.to_string()))?;
                let size = NonZeroU64::new(length).ok_or(ComputeError::InvalidArgument(
                    "a dispatched binding must span at least one byte",
                ))?;

                entries.push(wgpu::BindGroupEntry {
                    binding: binding.slot,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &binding.buffer.raw,
                        offset,
                        size: Some(size),
                    }),
                });
            }

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(kernel.entry_point.as_str()),
                layout: &kernel.pipeline.get_bind_group_layout(0),
                entries: &entries,
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(kernel.entry_point.as_str()),
                timestamp_writes: None,
            });
            pass.set_pipeline(&kernel.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(kernel.workgroups, 1, 1);
        }

        let submission = self.queue.submit(Some(encoder.finish()));

        self.take_error(ComputeError::Launch)?;

        Ok(WgpuReferenceEvent { submission })
    }

    fn wait(&self, event: &Self::Event) -> ComputeResult<()> {
        let _ = self.device.poll(wgpu::Maintain::WaitForSubmissionIndex(
            event.submission.clone(),
        ));
        Ok(())
    }

    fn synchronize(&self, _stream: &Self::Stream) -> ComputeResult<()> {
        let _ = self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! Generation tests. They need no device, so they run everywhere — which is
    //! what makes them worth having next to the lavapipe integration tests.

    use super::*;

    fn strides(dims: &[u64]) -> Vec<u32> {
        row_major_strides(dims).expect("representable strides")
    }

    #[test]
    fn row_major_strides_follow_the_reference_layout() {
        assert_eq!(strides(&[]), Vec::<u32>::new());
        assert_eq!(strides(&[5]), vec![1]);
        assert_eq!(strides(&[2, 3]), vec![3, 1]);
        assert_eq!(strides(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn dispatch_covers_every_element() {
        assert_eq!(dispatch_workgroups(0, 65535), Ok(0));
        assert_eq!(dispatch_workgroups(1, 65535), Ok(1));
        assert_eq!(dispatch_workgroups(64, 65535), Ok(1));
        assert_eq!(dispatch_workgroups(65, 65535), Ok(2));
        // Beyond the device limit the grid is capped; the shader strides.
        assert_eq!(dispatch_workgroups(64 * 100_000, 65535), Ok(65535));
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
    fn structural_opcodes_move_words_not_floats() {
        assert!(is_structural(ReferenceOpcode::ShapeCopy));
        assert!(is_structural(ReferenceOpcode::Permute));
        assert!(!is_structural(ReferenceOpcode::Add));
        assert!(!is_structural(ReferenceOpcode::Relu));
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
        let workgroups = dispatch_workgroups(elements, 65535).expect("dispatchable");
        generate_wgsl(artifact, &artifact.entry_point(), elements, workgroups)
            .expect("generated source")
    }

    /// Prints every generated shader. Run with `--nocapture` to review them.
    #[test]
    fn every_executable_opcode_generates_a_shader() {
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

            assert!(source.contains("@compute @workgroup_size(64)"));
            assert!(source.contains("fn scirust_reference_kernel_0("));
            assert!(source.contains("var<storage, read_write> result"));
            // The source must be a pure function of the artefact.
            assert_eq!(source, source_of(artifact), "{label} is not deterministic");
        }
    }

    #[test]
    fn a_zero_element_kernel_generates_no_body() {
        let artifact = artifact_of(Operation::Relu, 1, vec![0, 3], vec![0, 3]);
        let source = source_of(&artifact);

        println!("========== zero elements ==========\n{source}");
        assert!(!source.contains("loop"), "source: {source}");
        assert_eq!(
            dispatch_workgroups(0, 65535),
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
            source.contains("bitcast<f32>(0x80000000u)"),
            "source: {source}"
        );
    }

    #[test]
    fn entry_point_identifiers_are_validated() {
        assert!(valid_identifier("scirust_reference_kernel_0"));
        assert!(valid_identifier("_main"));
        assert!(!valid_identifier(""));
        assert!(!valid_identifier("0main"));
        assert!(!valid_identifier("main;"));
        assert!(!valid_identifier("main kernel"));
    }
}
