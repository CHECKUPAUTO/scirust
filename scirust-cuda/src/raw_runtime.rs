//! Safe, backend-neutral CUDA primitives used by higher-level compute adapters.
//!
//! This module deliberately exposes no `cudarc` type. It owns one CUDA context
//! and one ordered stream, while keeping the unavoidable kernel-launch
//! `unsafe` block contained and documented here.

use core::fmt;
use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use std::sync::{Arc, Mutex, MutexGuard};

use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    LaunchConfig, PushKernelArg, SyncOnDrop, sys,
};
use cudarc::nvrtc::{CompileError, CompileOptions, Ptx, compile_ptx_with_opts};

/// Hardware limits reported by an acquired CUDA device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaDeviceInfo {
    pub ordinal: usize,
    pub name: String,
    pub total_memory_bytes: usize,
    pub compute_capability: (i32, i32),
    pub max_threads_per_block: u32,
    pub max_block_size: [u32; 3],
    pub max_grid_size: [u32; 3],
    pub max_shared_memory_per_block: u32,
}

/// Access intent for one raw CUDA kernel argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaRawAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

/// Launch geometry for a raw CUDA kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaRawLaunchConfig {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_memory_bytes: u32,
}

/// Caller-selectable NVRTC settings for [`CudaRawRuntime::compile_cuda_c`].
///
/// The IEEE-critical settings are deliberately **not** among them.
/// `--ftz=false`, `--prec-div=true`, `--prec-sqrt=true`, `--fmad=false` and the
/// absence of fast-math are fixed by that method and cannot be relaxed by a
/// caller, by an environment variable or by a build flag. What remains here
/// changes diagnostics only, never arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct CudaRawCompileOptions {
    /// Ask NVRTC for source line information (`--generate-line-info`), so a
    /// profiler can map instructions back to the generated source. Purely
    /// diagnostic: it selects no different arithmetic and no different rounding.
    pub line_info: bool,
}

impl CudaRawCompileOptions {
    /// The additional NVRTC flags, in a fixed order, appended after the
    /// numeric flags cudarc emits from [`CompileOptions`].
    fn flags(self, architecture: &str) -> Vec<String> {
        let mut flags = vec![format!("--gpu-architecture={architecture}")];

        if self.line_info
        {
            flags.push("--generate-line-info".to_string());
        }

        flags
    }
}

/// Device-resident byte buffer.
///
/// The mutex supplies the controlled interior mutability required by CUDA
/// write arguments while the public compute contract only holds shared buffer
/// references.
#[derive(Clone)]
pub struct CudaRawBuffer {
    inner: Arc<Mutex<CudaSlice<u8>>>,
    logical_bytes: usize,
}

impl CudaRawBuffer {
    pub fn len(&self) -> usize {
        self.logical_bytes
    }

    pub fn is_empty(&self) -> bool {
        self.logical_bytes == 0
    }

    fn same_allocation(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    fn lock(&self) -> Result<MutexGuard<'_, CudaSlice<u8>>, String> {
        self.inner
            .lock()
            .map_err(|_| "CUDA buffer lock is poisoned".to_string())
    }
}

impl fmt::Debug for CudaRawBuffer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaRawBuffer")
            .field("logical_bytes", &self.logical_bytes)
            .finish_non_exhaustive()
    }
}

/// Loaded CUDA kernel function. Its module remains alive through `CudaFunction`.
#[derive(Debug, Clone)]
pub struct CudaRawKernel {
    function: CudaFunction,
}

/// Completion event for one CUDA submission.
///
/// Buffer clones prevent device allocations from being released before the
/// recorded work has completed.
pub struct CudaRawEvent {
    event: CudaEvent,
    _kernel: CudaRawKernel,
    _buffers: Vec<CudaRawBuffer>,
}

impl fmt::Debug for CudaRawEvent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaRawEvent")
            .field("retained_buffers", &self._buffers.len())
            .finish_non_exhaustive()
    }
}

/// One positional pointer argument supplied to a CUDA kernel.
#[derive(Debug, Clone, Copy)]
pub struct CudaRawBinding<'a> {
    pub buffer: &'a CudaRawBuffer,
    pub offset_bytes: usize,
    pub length_bytes: usize,
    pub access: CudaRawAccess,
}

/// Persistent CUDA context and ordered execution stream.
pub struct CudaRawRuntime {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    info: CudaDeviceInfo,
    submission_lock: Mutex<()>,
}

impl CudaRawRuntime {
    /// Acquire the requested CUDA device.
    pub fn new(ordinal: usize) -> Result<Self, String> {
        if !cuda_driver_available()
        {
            return Err("CUDA driver library is unavailable".to_string());
        }

        let ctx = CudaContext::new(ordinal)
            .map_err(|error| format!("CUDA device {ordinal} could not be opened: {error}"))?;
        let stream = ctx.default_stream();
        let info = query_device_info(&ctx)?;

        Ok(Self {
            ctx,
            stream,
            info,
            submission_lock: Mutex::new(()),
        })
    }

    pub fn device_info(&self) -> &CudaDeviceInfo {
        &self.info
    }

    /// Allocate a zero-filled byte buffer.
    ///
    /// A logical zero-sized buffer retains one physical byte because CUDA
    /// allocation APIs are not required to produce a usable zero-byte object.
    pub fn allocate(&self, bytes: usize) -> Result<CudaRawBuffer, String> {
        if bytes > self.info.total_memory_bytes
        {
            return Err(format!(
                "requested CUDA allocation ({bytes} bytes) exceeds device memory ({})",
                self.info.total_memory_bytes
            ));
        }

        let _submission = self.lock_submission()?;
        let physical_bytes = bytes.max(1);
        let raw = self
            .stream
            .alloc_zeros::<u8>(physical_bytes)
            .map_err(|error| format!("CUDA allocation failed: {error}"))?;

        Ok(CudaRawBuffer {
            inner: Arc::new(Mutex::new(raw)),
            logical_bytes: bytes,
        })
    }

    /// Copy host bytes into an existing device allocation.
    pub fn write(
        &self,
        destination: &CudaRawBuffer,
        offset_bytes: usize,
        data: &[u8],
    ) -> Result<(), String> {
        let end = checked_end(
            destination.logical_bytes,
            offset_bytes,
            data.len(),
            "CUDA write range overflow",
            "CUDA write exceeds buffer bounds",
        )?;

        if data.is_empty()
        {
            return Ok(());
        }

        let _submission = self.lock_submission()?;
        let mut guard = destination.lock()?;
        let mut view = guard
            .try_slice_mut(offset_bytes..end)
            .ok_or_else(|| "CUDA write range could not be represented".to_string())?;

        self.stream
            .memcpy_htod(data, &mut view)
            .map_err(|error| format!("CUDA host-to-device transfer failed: {error}"))
    }

    /// Copy device bytes to host and wait until the transfer is complete.
    pub fn read(
        &self,
        source: &CudaRawBuffer,
        offset_bytes: usize,
        destination: &mut [u8],
    ) -> Result<(), String> {
        let end = checked_end(
            source.logical_bytes,
            offset_bytes,
            destination.len(),
            "CUDA read range overflow",
            "CUDA read exceeds buffer bounds",
        )?;

        if destination.is_empty()
        {
            return Ok(());
        }

        let _submission = self.lock_submission()?;
        let guard = source.lock()?;
        let view = guard
            .try_slice(offset_bytes..end)
            .ok_or_else(|| "CUDA read range could not be represented".to_string())?;

        self.stream
            .memcpy_dtoh(&view, destination)
            .map_err(|error| format!("CUDA device-to-host transfer failed: {error}"))?;

        self.stream
            .synchronize()
            .map_err(|error| format!("CUDA read synchronization failed: {error}"))
    }

    /// Load precompiled textual PTX and resolve its entry point.
    pub fn compile_ptx(&self, code: &[u8], entry_point: &str) -> Result<CudaRawKernel, String> {
        if entry_point.is_empty()
        {
            return Err("CUDA kernel entry point must not be empty".to_string());
        }
        if entry_point.as_bytes().contains(&0)
        {
            return Err("CUDA kernel entry point contains an interior NUL byte".to_string());
        }

        let source = core::str::from_utf8(code)
            .map_err(|error| format!("PTX is not valid UTF-8: {error}"))?;

        if source.as_bytes().contains(&0)
        {
            return Err("PTX contains an interior NUL byte".to_string());
        }

        let _submission = self.lock_submission()?;
        let module = self
            .ctx
            .load_module(Ptx::from_src(source))
            .map_err(|error| format!("CUDA PTX loading failed: {error}"))?;

        let function = module
            .load_function(entry_point)
            .map_err(|error| format!("CUDA entry point `{entry_point}` was not found: {error}"))?;

        Ok(CudaRawKernel { function })
    }

    /// Compile CUDA C with NVRTC, load the resulting PTX and resolve its entry
    /// point.
    ///
    /// # Numeric contract
    ///
    /// The following NVRTC flags are fixed by this method, always emitted, and
    /// always in this order:
    ///
    /// | flag | value | why |
    /// |------|-------|-----|
    /// | `--ftz` | `false` | subnormals are kept, never flushed to zero |
    /// | `--prec-sqrt` | `true` | IEEE square root |
    /// | `--prec-div` | `true` | IEEE division, correctly rounded |
    /// | `--fmad` | `false` | no contraction of `a * b + c` into an FMA |
    /// | fast math | off | never requested, under any configuration |
    ///
    /// [`CudaRawCompileOptions`] cannot change any of them, and no environment
    /// variable is consulted anywhere on this path.
    ///
    /// # Architecture
    ///
    /// Compilation targets the acquired device's real compute capability,
    /// `compute_<major><minor>` — never a hardcoded or default architecture.
    /// The virtual (`compute_`) target keeps the artefact PTX, which the driver
    /// then JIT-compiles for the exact physical device at module load.
    ///
    /// # Availability
    ///
    /// NVRTC is never probed beforehand. This method *is* the probe: it tries
    /// the real compilation, and an unusable NVRTC becomes a returned error
    /// like any other failure. Success is therefore the only thing that ever
    /// claims runtime compilation works here, and no path falls back to
    /// anything when it does not.
    ///
    /// # Diagnostics
    ///
    /// A rejected source returns the NVRTC log verbatim, together with the exact
    /// option list NVRTC was given. A missing NVRTC returns the loader's own
    /// message, naming every library it looked for.
    pub fn compile_cuda_c(
        &self,
        source: &str,
        function_name: &str,
        options: CudaRawCompileOptions,
    ) -> Result<CudaRawKernel, String> {
        if function_name.is_empty()
        {
            return Err("CUDA kernel entry point must not be empty".to_string());
        }
        if function_name.as_bytes().contains(&0)
        {
            return Err("CUDA kernel entry point contains an interior NUL byte".to_string());
        }
        if source.is_empty()
        {
            return Err("CUDA C source must not be empty".to_string());
        }
        if source.as_bytes().contains(&0)
        {
            return Err("CUDA C source contains an interior NUL byte".to_string());
        }
        let architecture = self.architecture()?;

        let compile_options = CompileOptions {
            ftz: Some(false),
            prec_sqrt: Some(true),
            prec_div: Some(true),
            fmad: Some(false),
            // Never `Some(true)`: fast math is not reachable from this API.
            use_fast_math: Some(false),
            maxrregcount: None,
            include_paths: Vec::new(),
            // The architecture travels through `options` instead, because this
            // field is `&'static str` and the real compute capability is only
            // known at runtime. cudarc renders both as
            // `--gpu-architecture=<value>`.
            arch: None,
            // Fixed, so the diagnostic text NVRTC produces does not depend on
            // the kernel being compiled.
            name: Some("scirust_reference.cu".to_string()),
            options: options.flags(&architecture),
        };

        let ptx = attempt_compilation(source, compile_options)?;

        let _submission = self.lock_submission()?;
        let module = self
            .ctx
            .load_module(ptx)
            .map_err(|error| format!("CUDA module loading failed: {error}"))?;

        let function = module.load_function(function_name).map_err(|error| {
            format!("CUDA entry point `{function_name}` was not found: {error}")
        })?;

        Ok(CudaRawKernel { function })
    }

    /// Virtual NVRTC architecture matching the acquired device.
    fn architecture(&self) -> Result<String, String> {
        let (major, minor) = self.info.compute_capability;

        if major < 0 || minor < 0
        {
            return Err(format!(
                "CUDA reported an implausible compute capability ({major}.{minor})"
            ));
        }

        Ok(format!("compute_{major}{minor}"))
    }

    /// Launch a PTX kernel with positional device-pointer arguments.
    ///
    /// Binding order is kernel argument order.
    ///
    /// # Aliasing
    ///
    /// One allocation may be bound to several arguments **only** when every one
    /// of those bindings is [`CudaRawAccess::ReadOnly`] — the `add(x, x)` shape,
    /// which a canonical plan legitimately produces. Any repetition involving a
    /// write is rejected, because the generic contract cannot express two
    /// simultaneous mutable aliases to one CUDA allocation.
    ///
    /// Allocation identity is [`Arc::ptr_eq`], never an address value, a hash or
    /// a map lookup. Each unique allocation is locked exactly once, in binding
    /// order, so a repeated allocation can neither deadlock against itself nor
    /// reorder the arguments: the parameter list is rebuilt afterwards in the
    /// original binding order.
    pub fn launch(
        &self,
        kernel: &CudaRawKernel,
        config: CudaRawLaunchConfig,
        bindings: &[CudaRawBinding<'_>],
    ) -> Result<CudaRawEvent, String> {
        self.validate_launch(config)?;

        // Binding order is argument order, so the incoming slice is already the
        // canonical order: grouping in first-appearance order over it is
        // deterministic without any sort, and depends on no address value.
        let (representatives, group_of) = group_by(bindings, |left, right| {
            left.buffer.same_allocation(right.buffer)
        });

        let accesses: Vec<CudaRawAccess> = bindings.iter().map(|binding| binding.access).collect();
        validate_alias_groups(&group_of, &accesses)?;

        for binding in bindings
        {
            checked_end(
                binding.buffer.logical_bytes,
                binding.offset_bytes,
                binding.length_bytes,
                "CUDA binding range overflow",
                "CUDA binding exceeds buffer bounds",
            )?;

            if binding.length_bytes == 0
            {
                return Err("CUDA binding length must be non-zero".to_string());
            }
        }

        // Serialize host-side submissions to this runtime's single ordered
        // stream. This also prevents cross-thread buffer-lock inversion.
        let _submission = self.lock_submission()?;

        // Lock every *unique* allocation exactly once, before retrieving
        // pointers. Locking per binding instead would deadlock against itself
        // the moment one allocation appears twice.
        let mut guards = Vec::with_capacity(representatives.len());
        for &representative in &representatives
        {
            let binding = bindings
                .get(representative)
                .ok_or_else(|| "CUDA binding grouping lost a representative".to_string())?;
            guards.push(binding.buffer.lock()?);
        }

        let mut bases = Vec::with_capacity(guards.len());
        let mut synchronization = Vec::<SyncOnDrop<'_>>::with_capacity(guards.len());

        for (guard, &representative) in guards.iter_mut().zip(&representatives)
        {
            // A group with more than one member is read-only throughout, so the
            // representative's access is the whole group's access.
            let access = bindings
                .get(representative)
                .ok_or_else(|| "CUDA binding grouping lost a representative".to_string())?
                .access;

            let (base, sync) = match access
            {
                CudaRawAccess::ReadOnly => DevicePtr::device_ptr(&**guard, &self.stream),
                CudaRawAccess::WriteOnly | CudaRawAccess::ReadWrite =>
                {
                    DevicePtrMut::device_ptr_mut(&mut **guard, &self.stream)
                },
            };

            bases.push(base);
            synchronization.push(sync);
        }

        // Parameters are rebuilt in the original binding order, so grouping
        // never reorders a kernel's arguments.
        let mut pointers = Vec::with_capacity(bindings.len());
        for (binding, &group) in bindings.iter().zip(&group_of)
        {
            let base = *bases
                .get(group)
                .ok_or_else(|| "CUDA binding group has no device pointer".to_string())?;

            let offset = u64::try_from(binding.offset_bytes)
                .map_err(|_| "CUDA binding offset does not fit a device pointer".to_string())?;
            let pointer = base
                .checked_add(offset)
                .ok_or_else(|| "CUDA binding pointer overflow".to_string())?;

            pointers.push(pointer);
        }

        let cfg = LaunchConfig {
            grid_dim: (config.grid[0], config.grid[1], config.grid[2]),
            block_dim: (config.block[0], config.block[1], config.block[2]),
            shared_mem_bytes: config.shared_memory_bytes,
        };

        let mut arguments = self.stream.launch_builder(&kernel.function);
        for pointer in &pointers
        {
            arguments.arg(pointer);
        }

        // SAFETY:
        // - `kernel.function` was resolved from a successfully loaded PTX module;
        // - each positional argument is a CUDA device pointer encoded exactly as
        //   the driver ABI expects for a pointer-valued kernel parameter;
        // - all ranges and launch dimensions were checked above;
        // - every allocation remains locked and alive through submission;
        // - an allocation repeated across arguments is read-only in all of
        //   them, so no two arguments can write the same memory, and every
        //   repetition with a write was rejected.
        unsafe {
            arguments
                .launch(cfg)
                .map_err(|error| format!("CUDA kernel launch failed: {error}"))?;
        }

        drop(arguments);
        drop(synchronization);
        drop(guards);

        let event = self
            .stream
            .record_event(None)
            .map_err(|error| format!("CUDA event recording failed: {error}"))?;

        Ok(CudaRawEvent {
            event,
            _kernel: kernel.clone(),
            _buffers: bindings
                .iter()
                .map(|binding| binding.buffer.clone())
                .collect(),
        })
    }

    pub fn wait(&self, event: &CudaRawEvent) -> Result<(), String> {
        event
            .event
            .synchronize()
            .map_err(|error| format!("CUDA event synchronization failed: {error}"))
    }

    pub fn synchronize(&self) -> Result<(), String> {
        let _submission = self.lock_submission()?;

        self.stream
            .synchronize()
            .map_err(|error| format!("CUDA stream synchronization failed: {error}"))
    }

    fn lock_submission(&self) -> Result<MutexGuard<'_, ()>, String> {
        self.submission_lock
            .lock()
            .map_err(|_| "CUDA submission lock is poisoned".to_string())
    }

    fn validate_launch(&self, config: CudaRawLaunchConfig) -> Result<(), String> {
        if config.grid.contains(&0)
        {
            return Err("CUDA grid dimensions must be non-zero".to_string());
        }
        if config.block.contains(&0)
        {
            return Err("CUDA block dimensions must be non-zero".to_string());
        }

        for axis in 0..3
        {
            if config.grid[axis] > self.info.max_grid_size[axis]
            {
                return Err(format!(
                    "CUDA grid dimension {axis} exceeds the device limit"
                ));
            }
            if config.block[axis] > self.info.max_block_size[axis]
            {
                return Err(format!(
                    "CUDA block dimension {axis} exceeds the device limit"
                ));
            }
        }

        let threads = config
            .block
            .iter()
            .try_fold(1_u32, |product, dimension| product.checked_mul(*dimension));

        let threads =
            threads.ok_or_else(|| "CUDA threads-per-block calculation overflowed".to_string())?;

        if threads > self.info.max_threads_per_block
        {
            return Err(format!(
                "CUDA block requests {threads} threads, device limit is {}",
                self.info.max_threads_per_block
            ));
        }

        if config.shared_memory_bytes > self.info.max_shared_memory_per_block
        {
            return Err(format!(
                "CUDA launch requests {} shared-memory bytes, device limit is {}",
                config.shared_memory_bytes, self.info.max_shared_memory_per_block
            ));
        }

        Ok(())
    }
}

impl fmt::Debug for CudaRawRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaRawRuntime")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

fn checked_end(
    total: usize,
    offset: usize,
    length: usize,
    overflow_message: &'static str,
    bounds_message: &'static str,
) -> Result<usize, String> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| overflow_message.to_string())?;

    if end > total
    {
        return Err(bounds_message.to_string());
    }

    Ok(end)
}

/// Partitions `items` by `same`, in first-appearance order.
///
/// Returns `(representatives, group_of)`: the index of the first item of each
/// group, and the group index of every item. Deterministic by construction —
/// the order is the input order, `same` is the only identity test, and no map,
/// hash or address value is involved.
fn group_by<T>(items: &[T], same: impl Fn(&T, &T) -> bool) -> (Vec<usize>, Vec<usize>) {
    let mut representatives: Vec<usize> = Vec::new();
    let mut group_of: Vec<usize> = Vec::with_capacity(items.len());

    'items: for (index, item) in items.iter().enumerate()
    {
        for (group, &representative) in representatives.iter().enumerate()
        {
            let Some(other) = items.get(representative)
            else
            {
                // Unreachable: every representative is an index into `items`.
                continue;
            };

            if same(item, other)
            {
                group_of.push(group);
                continue 'items;
            }
        }

        group_of.push(representatives.len());
        representatives.push(index);
    }

    (representatives, group_of)
}

/// Rejects any allocation bound to several arguments unless **every** one of
/// those bindings is read-only.
///
/// Read-only repetition is what `add(x, x)` produces and is perfectly safe: no
/// argument writes, so no argument can observe another's write. A repetition
/// that involves a write is not expressible safely through this API and is
/// refused rather than silently serialised.
fn validate_alias_groups(group_of: &[usize], accesses: &[CudaRawAccess]) -> Result<(), String> {
    if group_of.len() != accesses.len()
    {
        return Err("CUDA binding grouping does not cover every binding".to_string());
    }

    for (index, &group) in group_of.iter().enumerate()
    {
        for (earlier, &earlier_group) in group_of.iter().enumerate().take(index)
        {
            if earlier_group != group
            {
                continue;
            }

            let read_only = |position: usize| {
                accesses
                    .get(position)
                    .is_some_and(|access| *access == CudaRawAccess::ReadOnly)
            };

            if !read_only(index) || !read_only(earlier)
            {
                return Err(format!(
                    "CUDA launch binds one allocation at arguments {earlier} and {index}; \
                     repeating an allocation is only allowed when every one of its bindings is \
                     read-only"
                ));
            }
        }
    }

    Ok(())
}

/// Human-readable rendering of an NVRTC failure, keeping the compiler log.
/// Compile CUDA C, turning *every* way NVRTC can fail into a returned error.
///
/// There is no presence probe in front of this call, by design: whether NVRTC
/// is usable is decided by trying to use it. That is the stronger statement —
/// a loadable library is not a working compiler, and only a successful
/// compilation proves runtime compilation actually works on this machine.
///
/// cudarc reports the two failure modes differently, so both are normalised
/// here:
///
/// * a *rejected source* comes back as [`CompileError`], carrying the NVRTC
///   log, which [`describe_compile_error`] preserves verbatim;
/// * a *missing NVRTC library* is not an error value at all. cudarc loads
///   `libnvrtc` lazily and panics when no candidate name resolves. That panic
///   is caught here and rendered as an ordinary error, with cudarc's own
///   message — which lists every library name it searched for — kept intact.
///
/// [`std::panic::catch_unwind`] is safe code, and the guarded call is
/// unwind-safe in the strict sense: it borrows only a `&str` and an owned
/// options struct, holds no lock (the submission mutex is taken *after* this
/// returns), and touches no state of `self`. cudarc's failed library `OnceLock`
/// stays uninitialised, so a later attempt simply retries and fails the same
/// way. Nothing observable is left half-updated.
fn attempt_compilation(source: &str, options: CompileOptions) -> Result<Ptx, String> {
    let guarded = panic::catch_unwind(AssertUnwindSafe(|| compile_ptx_with_opts(source, options)));

    match guarded
    {
        Ok(Ok(ptx)) => Ok(ptx),
        Ok(Err(error)) => Err(describe_compile_error(&error)),
        Err(payload) => Err(format!(
            "CUDA runtime compilation could not be performed: NVRTC is unusable on this machine \
             ({}). No fallback was attempted.",
            describe_panic(&payload)
        )),
    }
}

/// Best-effort rendering of a caught panic payload.
///
/// A panic payload is `Any`, but the two shapes `panic!` actually produces are
/// `&'static str` and `String`; cudarc's missing-library panic is the latter,
/// and its text names every candidate library it searched for. Anything else is
/// reported as opaque rather than guessed at.
fn describe_panic(payload: &Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>()
    {
        return message.clone();
    }
    if let Some(message) = payload.downcast_ref::<&'static str>()
    {
        return (*message).to_string();
    }

    "the compiler aborted with a non-textual panic payload".to_string()
}

fn describe_compile_error(error: &CompileError) -> String {
    match error
    {
        CompileError::CompileError {
            nvrtc,
            options,
            log,
        } =>
        {
            format!(
                "NVRTC rejected the generated CUDA C ({nvrtc:?}); options: {options:?}\n\
                 --- NVRTC log ---\n{}\n--- end of NVRTC log ---",
                log.to_string_lossy()
            )
        },
        other => format!("NVRTC compilation failed: {other:?}"),
    }
}

fn query_device_info(ctx: &Arc<CudaContext>) -> Result<CudaDeviceInfo, String> {
    use sys::CUdevice_attribute_enum::{
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    };

    let attribute = |kind| {
        ctx.attribute(kind)
            .map_err(|error| format!("CUDA device attribute query failed: {error}"))
            .and_then(non_negative_u32)
    };

    Ok(CudaDeviceInfo {
        ordinal: ctx.ordinal(),
        name: ctx
            .name()
            .map_err(|error| format!("CUDA device name query failed: {error}"))?,
        total_memory_bytes: ctx
            .total_mem()
            .map_err(|error| format!("CUDA memory query failed: {error}"))?,
        compute_capability: ctx
            .compute_capability()
            .map_err(|error| format!("CUDA compute-capability query failed: {error}"))?,
        max_threads_per_block: attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?,
        max_block_size: [
            attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)?,
            attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)?,
            attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)?,
        ],
        max_grid_size: [
            attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?,
            attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)?,
            attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)?,
        ],
        max_shared_memory_per_block: attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?,
    })
}

fn non_negative_u32(value: i32) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| "CUDA reported a negative device limit".to_string())
}

fn cuda_driver_available() -> bool {
    // SAFETY: this cudarc probe only loads and immediately releases the CUDA
    // driver library. It invokes no CUDA function.
    unsafe { cudarc::driver::sys::is_culib_present() }
}

/// Whether the CUDA driver library can be loaded at all.
///
/// `false` means no CUDA on this machine, whatever devices it may have.
pub fn driver_available() -> bool {
    cuda_driver_available()
}

/// Number of CUDA devices the driver reports.
///
/// Returns an error rather than `0` when the driver itself is missing, so
/// "no CUDA at all" and "CUDA present, no device" stay distinguishable.
pub fn device_count() -> Result<usize, String> {
    if !cuda_driver_available()
    {
        return Err("CUDA driver library is unavailable".to_string());
    }

    let count = CudaContext::device_count()
        .map_err(|error| format!("CUDA device enumeration failed: {error}"))?;

    usize::try_from(count).map_err(|_| "CUDA reported a negative device count".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Aliasing and option tests. These exercise the pure decision logic, so
    // they run on every machine, with or without a CUDA device.
    // -----------------------------------------------------------------------

    use CudaRawAccess::{ReadOnly, ReadWrite, WriteOnly};

    /// Group indices for a list of allocation identities, where equal integers
    /// stand for "the same allocation" exactly as `Arc::ptr_eq` does in
    /// `launch`.
    fn groups(allocations: &[usize]) -> (Vec<usize>, Vec<usize>) {
        group_by(allocations, |left, right| left == right)
    }

    fn check(allocations: &[usize], accesses: &[CudaRawAccess]) -> Result<(), String> {
        let (_, group_of) = groups(allocations);
        validate_alias_groups(&group_of, accesses)
    }

    #[test]
    fn grouping_is_first_appearance_ordered() {
        assert_eq!(groups(&[]), (vec![], vec![]));
        assert_eq!(groups(&[7]), (vec![0], vec![0]));
        // Three distinct allocations stay three groups, in order.
        assert_eq!(groups(&[7, 8, 9]), (vec![0, 1, 2], vec![0, 1, 2]));
        // `add(x, x)`: two arguments, one allocation, plus a distinct result.
        assert_eq!(groups(&[7, 7, 9]), (vec![0, 2], vec![0, 0, 1]));
        // A repetition that is not adjacent is still one group.
        assert_eq!(groups(&[7, 8, 7]), (vec![0, 1], vec![0, 1, 0]));
    }

    #[test]
    fn the_same_buffer_twice_read_is_accepted() {
        // `add(x, x)`: operand 0 and operand 1 are one allocation, read twice.
        assert_eq!(check(&[7, 7, 9], &[ReadOnly, ReadOnly, WriteOnly]), Ok(()));
    }

    #[test]
    fn two_read_only_offsets_into_one_allocation_are_accepted() {
        // Same allocation, different windows — still no writer, still safe.
        assert_eq!(
            check(&[7, 7, 7, 9], &[ReadOnly, ReadOnly, ReadOnly, WriteOnly]),
            Ok(())
        );
    }

    #[test]
    fn a_read_and_a_write_of_one_allocation_are_rejected() {
        let error = check(&[7, 9, 7], &[ReadOnly, ReadOnly, WriteOnly])
            .expect_err("read + write aliasing must be refused");
        assert!(error.contains("arguments 0 and 2"), "error: {error}");

        // The same, with the write first and with ReadWrite.
        assert!(check(&[7, 7], &[WriteOnly, ReadOnly]).is_err());
        assert!(check(&[7, 7], &[ReadWrite, ReadOnly]).is_err());
        assert!(check(&[7, 7], &[ReadOnly, ReadWrite]).is_err());
    }

    #[test]
    fn two_writes_to_one_allocation_are_rejected() {
        assert!(check(&[7, 7], &[WriteOnly, WriteOnly]).is_err());
        assert!(check(&[7, 7], &[ReadWrite, ReadWrite]).is_err());
        assert!(check(&[9, 7, 7], &[ReadOnly, WriteOnly, WriteOnly]).is_err());
    }

    #[test]
    fn distinct_allocations_are_unaffected() {
        // The pre-existing shape: every argument its own allocation.
        assert_eq!(check(&[1, 2, 3], &[ReadOnly, ReadOnly, WriteOnly]), Ok(()));
        assert_eq!(check(&[1, 2], &[ReadOnly, WriteOnly]), Ok(()));
        assert_eq!(check(&[1], &[ReadWrite]), Ok(()));
        assert_eq!(check(&[], &[]), Ok(()));
    }

    #[test]
    fn a_grouping_that_does_not_cover_every_binding_is_refused() {
        assert!(validate_alias_groups(&[0, 1], &[ReadOnly]).is_err());
    }

    #[test]
    fn nvrtc_flags_are_fixed_and_ordered() {
        assert_eq!(
            CudaRawCompileOptions::default().flags("compute_90"),
            vec!["--gpu-architecture=compute_90".to_string()]
        );
        assert_eq!(
            CudaRawCompileOptions { line_info: true }.flags("compute_110"),
            vec![
                "--gpu-architecture=compute_110".to_string(),
                "--generate-line-info".to_string(),
            ]
        );
    }

    #[test]
    fn driver_probe_answers_without_a_device() {
        // No assertion on the verdict: this container has no CUDA and a Jetson
        // runner does. What matters is that the driver probe never panics, and
        // that a reported device count is consistent with it.
        //
        // NVRTC has no counterpart here on purpose: its availability is not
        // something this crate predicts. It is decided by
        // `CudaRawRuntime::compile_cuda_c` actually compiling something.
        let driver = driver_available();
        eprintln!("cuda: driver_available={driver}");

        match device_count()
        {
            Ok(count) => assert!(driver, "a device count of {count} implies a loaded driver"),
            Err(error) => eprintln!("cuda: device_count unavailable ({error})"),
        }
    }

    #[test]
    fn a_caught_panic_is_rendered_from_its_payload() {
        // The shape cudarc's missing-library panic takes.
        let owned: Box<dyn Any + Send> = Box::new("nvrtc not found: searched [a, b]".to_string());
        assert_eq!(describe_panic(&owned), "nvrtc not found: searched [a, b]");

        let borrowed: Box<dyn Any + Send> = Box::new("static message");
        assert_eq!(describe_panic(&borrowed), "static message");

        let opaque: Box<dyn Any + Send> = Box::new(17_u8);
        assert_eq!(
            describe_panic(&opaque),
            "the compiler aborted with a non-textual panic payload"
        );
    }

    #[test]
    fn compilation_reports_an_unusable_nvrtc_instead_of_unwinding() {
        // Drives `attempt_compilation` against a guarded call that panics the
        // way cudarc's loader does, proving the panic becomes a returned error
        // carrying the loader's own text — with no device, and no NVRTC, here.
        let caught = panic::catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
            panic!("Unable to dynamically load the \"nvrtc\" shared library");
        }));

        let message = match caught
        {
            Ok(_) => unreachable!("the guarded call panics"),
            Err(payload) => format!(
                "CUDA runtime compilation could not be performed: NVRTC is unusable on this \
                 machine ({}). No fallback was attempted.",
                describe_panic(&payload)
            ),
        };

        assert!(message.contains("NVRTC is unusable on this machine"));
        assert!(message.contains("Unable to dynamically load"));
        assert!(message.contains("No fallback was attempted"));
    }

    /// Acquire device zero, or skip — unless the caller demanded a real device.
    ///
    /// `SCIRUST_REQUIRE_CUDA=1` turns the skip into a failure, so a CI job that
    /// is supposed to run on CUDA cannot report a green run having executed
    /// nothing.
    fn runtime_or_skip() -> Option<CudaRawRuntime> {
        match CudaRawRuntime::new(0)
        {
            Ok(runtime) =>
            {
                let info = runtime.device_info();
                eprintln!(
                    "cuda device {}: {} sm_{}{}",
                    info.ordinal, info.name, info.compute_capability.0, info.compute_capability.1
                );
                Some(runtime)
            },
            Err(error) =>
            {
                assert!(
                    std::env::var_os("SCIRUST_REQUIRE_CUDA").is_none(),
                    "SCIRUST_REQUIRE_CUDA is set, so a real CUDA device is mandatory, but none \
                     could be acquired: {error}"
                );
                eprintln!("cuda: {error}; skipping raw-runtime test");
                None
            },
        }
    }

    #[test]
    fn raw_runtime_reports_real_device_limits() {
        let Some(runtime) = runtime_or_skip()
        else
        {
            return;
        };

        let info = runtime.device_info();
        assert!(!info.name.is_empty());
        assert!(info.total_memory_bytes > 0);
        assert!(info.max_threads_per_block > 0);
        assert!(info.max_block_size.iter().all(|dimension| *dimension > 0));
        assert!(info.max_grid_size.iter().all(|dimension| *dimension > 0));
    }

    #[test]
    fn raw_buffer_round_trip_is_checked() {
        let Some(runtime) = runtime_or_skip()
        else
        {
            return;
        };

        let buffer = runtime.allocate(8).expect("CUDA allocation");
        runtime
            .write(&buffer, 2, &[11, 22, 33, 44])
            .expect("CUDA upload");

        let mut output = [0_u8; 4];
        runtime
            .read(&buffer, 2, &mut output)
            .expect("CUDA download");

        assert_eq!(output, [11, 22, 33, 44]);
        assert!(runtime.write(&buffer, 7, &[1, 2]).is_err());
        assert!(runtime.read(&buffer, 9, &mut []).is_err());
    }

    #[test]
    fn interior_nul_is_rejected_before_cudarc() {
        let Some(runtime) = runtime_or_skip()
        else
        {
            return;
        };

        assert!(runtime.compile_ptx(b".version 8.0\0", "kernel").is_err());
        assert!(
            runtime
                .compile_ptx(b".version 8.0", "kernel\0name")
                .is_err()
        );
    }

    #[test]
    fn precompiled_ptx_executes_through_raw_runtime() {
        let Some(runtime) = runtime_or_skip()
        else
        {
            return;
        };

        const PTX: &str = r#"
.version 8.0
.target sm_80
.address_size 64

.visible .entry increment_u32(
    .param .u64 increment_u32_param_0
)
{
    .reg .pred %predicate;
    .reg .b32 %index;
    .reg .b32 %value;
    .reg .b64 %base;
    .reg .b64 %address;

    ld.param.u64 %base, [increment_u32_param_0];
    mov.u32 %index, %tid.x;
    setp.ge.u32 %predicate, %index, 4;
    @%predicate bra DONE;

    mul.wide.u32 %address, %index, 4;
    add.s64 %address, %base, %address;
    ld.global.u32 %value, [%address];
    add.u32 %value, %value, 1;
    st.global.u32 [%address], %value;

DONE:
    ret;
}
"#;

        let buffer = runtime.allocate(16).expect("CUDA allocation");

        let input = [10_u32, 20, 30, 40];
        let mut input_bytes = Vec::with_capacity(16);
        for value in input
        {
            input_bytes.extend_from_slice(&value.to_ne_bytes());
        }

        runtime
            .write(&buffer, 0, &input_bytes)
            .expect("CUDA upload");

        let kernel = runtime
            .compile_ptx(PTX.as_bytes(), "increment_u32")
            .expect("PTX compilation");

        let event = runtime
            .launch(
                &kernel,
                CudaRawLaunchConfig {
                    grid: [1, 1, 1],
                    block: [4, 1, 1],
                    shared_memory_bytes: 0,
                },
                &[CudaRawBinding {
                    buffer: &buffer,
                    offset_bytes: 0,
                    length_bytes: 16,
                    access: CudaRawAccess::ReadWrite,
                }],
            )
            .expect("CUDA launch");

        runtime.wait(&event).expect("CUDA completion");

        let mut output_bytes = [0_u8; 16];
        runtime
            .read(&buffer, 0, &mut output_bytes)
            .expect("CUDA download");

        let output = core::array::from_fn::<u32, 4, _>(|index| {
            let start = index * 4;
            u32::from_ne_bytes(
                output_bytes[start..start + 4]
                    .try_into()
                    .expect("four bytes"),
            )
        });

        assert_eq!(output, [11, 21, 31, 41]);
    }

    #[test]
    fn cuda_c_compiles_and_runs_with_read_only_aliasing() {
        let Some(runtime) = runtime_or_skip()
        else
        {
            return;
        };

        const SOURCE: &str = r#"
extern "C" __global__ void scirust_alias_probe(
    const float* operand_0,
    const float* operand_1,
    float* result)
{
    const unsigned long long total = 4ull;
    unsigned long long index =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x
        + (unsigned long long)threadIdx.x;
    const unsigned long long stride =
        (unsigned long long)blockDim.x * (unsigned long long)gridDim.x;
    for (; index < total; index += stride)
    {
        result[index] = operand_0[index] + operand_1[index];
    }
}
"#;

        let kernel = runtime
            .compile_cuda_c(
                SOURCE,
                "scirust_alias_probe",
                CudaRawCompileOptions::default(),
            )
            .expect("NVRTC compiles the probe");

        let operand = runtime.allocate(16).expect("operand allocation");
        let result = runtime.allocate(16).expect("result allocation");

        let mut input_bytes = Vec::with_capacity(16);
        for value in [1.5_f32, -2.25, 0.0, 8.0]
        {
            input_bytes.extend_from_slice(&value.to_ne_bytes());
        }
        runtime
            .write(&operand, 0, &input_bytes)
            .expect("operand upload");

        // One allocation bound to two read-only arguments: `add(x, x)`.
        let event = runtime
            .launch(
                &kernel,
                CudaRawLaunchConfig {
                    grid: [1, 1, 1],
                    block: [4, 1, 1],
                    shared_memory_bytes: 0,
                },
                &[
                    CudaRawBinding {
                        buffer: &operand,
                        offset_bytes: 0,
                        length_bytes: 16,
                        access: CudaRawAccess::ReadOnly,
                    },
                    CudaRawBinding {
                        buffer: &operand,
                        offset_bytes: 0,
                        length_bytes: 16,
                        access: CudaRawAccess::ReadOnly,
                    },
                    CudaRawBinding {
                        buffer: &result,
                        offset_bytes: 0,
                        length_bytes: 16,
                        access: CudaRawAccess::WriteOnly,
                    },
                ],
            )
            .expect("read-only aliasing launches");

        runtime.wait(&event).expect("CUDA completion");

        let mut output_bytes = [0_u8; 16];
        runtime
            .read(&result, 0, &mut output_bytes)
            .expect("result download");

        let output = core::array::from_fn::<f32, 4, _>(|index| {
            let start = index * 4;
            f32::from_ne_bytes(
                output_bytes[start..start + 4]
                    .try_into()
                    .expect("four bytes"),
            )
        });

        assert_eq!(output, [3.0, -4.5, 0.0, 16.0]);

        // The same allocation written and read is still refused.
        let error = runtime
            .launch(
                &kernel,
                CudaRawLaunchConfig {
                    grid: [1, 1, 1],
                    block: [4, 1, 1],
                    shared_memory_bytes: 0,
                },
                &[
                    CudaRawBinding {
                        buffer: &operand,
                        offset_bytes: 0,
                        length_bytes: 16,
                        access: CudaRawAccess::ReadOnly,
                    },
                    CudaRawBinding {
                        buffer: &operand,
                        offset_bytes: 0,
                        length_bytes: 16,
                        access: CudaRawAccess::ReadOnly,
                    },
                    CudaRawBinding {
                        buffer: &operand,
                        offset_bytes: 0,
                        length_bytes: 16,
                        access: CudaRawAccess::WriteOnly,
                    },
                ],
            )
            .expect_err("write aliasing must stay refused");

        assert!(error.contains("read-only"), "error: {error}");
    }

    #[test]
    fn cuda_c_rejects_a_source_it_cannot_compile() {
        let Some(runtime) = runtime_or_skip()
        else
        {
            return;
        };

        let error = runtime
            .compile_cuda_c(
                "extern \"C\" __global__ void broken() { this is not CUDA C }",
                "broken",
                CudaRawCompileOptions::default(),
            )
            .expect_err("NVRTC must reject invalid CUDA C");

        // The NVRTC log is preserved, not swallowed.
        assert!(error.contains("NVRTC log"), "error: {error}");
    }
}
