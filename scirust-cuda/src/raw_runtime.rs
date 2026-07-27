//! Safe, backend-neutral CUDA primitives used by higher-level compute adapters.
//!
//! This module deliberately exposes no `cudarc` type. It owns one CUDA context
//! and one ordered stream, while keeping the unavoidable kernel-launch
//! `unsafe` block contained and documented here.

use core::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    LaunchConfig, PushKernelArg, SyncOnDrop, sys,
};
use cudarc::nvrtc::Ptx;

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

    /// Launch a PTX kernel with positional device-pointer arguments.
    ///
    /// Binding order is kernel argument order. Duplicate allocations are
    /// rejected because the generic contract cannot safely express two
    /// simultaneous mutable aliases to one CUDA allocation.
    pub fn launch(
        &self,
        kernel: &CudaRawKernel,
        config: CudaRawLaunchConfig,
        bindings: &[CudaRawBinding<'_>],
    ) -> Result<CudaRawEvent, String> {
        self.validate_launch(config)?;
        validate_unique_buffers(bindings)?;

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

        // Lock every unique allocation before retrieving pointers.
        let mut guards = Vec::with_capacity(bindings.len());
        for binding in bindings
        {
            guards.push(binding.buffer.lock()?);
        }

        let mut pointers = Vec::with_capacity(bindings.len());
        let mut synchronization = Vec::<SyncOnDrop<'_>>::with_capacity(bindings.len());

        for (guard, binding) in guards.iter_mut().zip(bindings)
        {
            let (base, sync) = match binding.access
            {
                CudaRawAccess::ReadOnly => DevicePtr::device_ptr(&**guard, &self.stream),
                CudaRawAccess::WriteOnly | CudaRawAccess::ReadWrite =>
                {
                    DevicePtrMut::device_ptr_mut(&mut **guard, &self.stream)
                },
            };

            let offset = u64::try_from(binding.offset_bytes)
                .map_err(|_| "CUDA binding offset does not fit a device pointer".to_string())?;
            let pointer = base
                .checked_add(offset)
                .ok_or_else(|| "CUDA binding pointer overflow".to_string())?;

            pointers.push(pointer);
            synchronization.push(sync);
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
        // - duplicate allocations are rejected, preventing conflicting aliases.
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

fn validate_unique_buffers(bindings: &[CudaRawBinding<'_>]) -> Result<(), String> {
    for (index, binding) in bindings.iter().enumerate()
    {
        if bindings[..index]
            .iter()
            .any(|previous| binding.buffer.same_allocation(previous.buffer))
        {
            return Err("CUDA launch cannot bind the same allocation more than once".to_string());
        }
    }

    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime_or_skip() -> Option<CudaRawRuntime> {
        match CudaRawRuntime::new(0)
        {
            Ok(runtime) => Some(runtime),
            Err(error) =>
            {
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
}
