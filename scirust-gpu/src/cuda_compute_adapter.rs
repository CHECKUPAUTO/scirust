//! `scirust-compute` adapter for the CUDA raw runtime.

use core::fmt;

use scirust_compute::{
    BufferAccess, BufferBinding, ComputeBackend, ComputeError, ComputeResult, DType,
    DeviceCapabilities, DeviceId, DeviceKind, KernelFormat, KernelModule, LaunchConfig,
    MemorySpace,
};
use scirust_cuda::{
    CudaRawAccess, CudaRawBinding, CudaRawBuffer, CudaRawEvent, CudaRawKernel, CudaRawLaunchConfig,
    CudaRawRuntime,
};

/// Persistent CUDA implementation of [`ComputeBackend`].
pub struct CudaComputeAdapter {
    runtime: CudaRawRuntime,
    capabilities: DeviceCapabilities,
}

impl CudaComputeAdapter {
    /// Acquire CUDA device zero.
    pub fn new() -> ComputeResult<Self> {
        let runtime = CudaRawRuntime::new(0)
            .map_err(|_| ComputeError::BackendUnavailable(DeviceKind::Cuda))?;

        Ok(Self::from_runtime(runtime))
    }

    /// Wrap an already acquired raw CUDA runtime.
    pub fn from_runtime(runtime: CudaRawRuntime) -> Self {
        let info = runtime.device_info();

        let capabilities = DeviceCapabilities {
            device: DeviceId::new(
                DeviceKind::Cuda,
                u32::try_from(info.ordinal).unwrap_or(u32::MAX),
            ),
            name: format!(
                "scirust-gpu-cuda: {} (sm_{}{})",
                info.name, info.compute_capability.0, info.compute_capability.1
            ),
            supported_dtypes: vec![
                DType::U8,
                DType::I8,
                DType::U16,
                DType::I16,
                DType::F16,
                DType::Bf16,
                DType::U32,
                DType::I32,
                DType::F32,
                DType::U64,
                DType::I64,
                DType::F64,
            ],
            max_buffer_bytes: Some(info.total_memory_bytes),
            max_workgroup_size: info.max_block_size,
            supports_async_execution: true,
        };

        Self {
            runtime,
            capabilities,
        }
    }

    /// Underlying ordered CUDA runtime.
    pub fn runtime(&self) -> &CudaRawRuntime {
        &self.runtime
    }

    /// Capabilities reported by the acquired CUDA device.
    pub fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }
}

impl fmt::Debug for CudaComputeAdapter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaComputeAdapter")
            .field("capabilities", &self.capabilities)
            .finish_non_exhaustive()
    }
}

/// Device-resident CUDA byte buffer.
#[derive(Clone)]
pub struct CudaComputeBuffer {
    raw: CudaRawBuffer,
    alignment: usize,
    memory_space: MemorySpace,
}

impl CudaComputeBuffer {
    pub fn len(&self) -> usize {
        self.raw.len()
    }

    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }

    pub fn alignment(&self) -> usize {
        self.alignment
    }

    pub fn memory_space(&self) -> MemorySpace {
        self.memory_space
    }
}

impl fmt::Debug for CudaComputeBuffer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaComputeBuffer")
            .field("bytes", &self.len())
            .field("alignment", &self.alignment)
            .field("memory_space", &self.memory_space)
            .finish_non_exhaustive()
    }
}

/// Loaded precompiled PTX kernel.
#[derive(Debug, Clone)]
pub struct CudaComputeKernel {
    raw: CudaRawKernel,
}

/// Logical stream backed by the runtime's single ordered CUDA stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaComputeStream(());

/// Completion event returned by a CUDA kernel launch.
#[derive(Debug)]
pub struct CudaComputeEvent {
    raw: CudaRawEvent,
}

impl ComputeBackend for CudaComputeAdapter {
    type Buffer = CudaComputeBuffer;
    type Kernel = CudaComputeKernel;
    type Stream = CudaComputeStream;
    type Event = CudaComputeEvent;

    fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }

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

        // The raw contract currently proves byte alignment only. Stronger
        // alignment requests will be enabled once their CUDA guarantee is
        // represented and tested explicitly.
        if alignment != 1
        {
            return Err(ComputeError::Unsupported(
                "CUDA compute buffers currently support byte alignment only",
            ));
        }

        if memory_space != MemorySpace::Device
        {
            return Err(ComputeError::Unsupported(
                "CUDA compute adapter currently supports device memory only",
            ));
        }

        let raw = self
            .runtime
            .allocate(bytes)
            .map_err(ComputeError::Allocation)?;

        Ok(CudaComputeBuffer {
            raw,
            alignment,
            memory_space,
        })
    }

    fn write(
        &self,
        destination: &Self::Buffer,
        offset_bytes: usize,
        data: &[u8],
    ) -> ComputeResult<()> {
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
        self.runtime
            .read(&source.raw, offset_bytes, destination)
            .map_err(ComputeError::Transfer)
    }

    fn compile(&self, module: &KernelModule) -> ComputeResult<Self::Kernel> {
        if module.format != KernelFormat::Ptx
        {
            return Err(ComputeError::Unsupported(
                "CUDA compute adapter accepts precompiled PTX only",
            ));
        }

        let raw = self
            .runtime
            .compile_ptx(&module.code, &module.entry_point)
            .map_err(ComputeError::Compilation)?;

        Ok(CudaComputeKernel { raw })
    }

    fn create_stream(&self) -> ComputeResult<Self::Stream> {
        Ok(CudaComputeStream(()))
    }

    fn launch(
        &self,
        kernel: &Self::Kernel,
        _stream: &Self::Stream,
        config: LaunchConfig,
        bindings: &[BufferBinding<'_, Self::Buffer>],
    ) -> ComputeResult<Self::Event> {
        let mut ordered = bindings.iter().collect::<Vec<_>>();
        ordered.sort_by_key(|binding| binding.slot);

        if ordered.windows(2).any(|pair| pair[0].slot == pair[1].slot)
        {
            return Err(ComputeError::InvalidArgument(
                "buffer binding slots must be unique",
            ));
        }

        let raw_bindings = ordered
            .iter()
            .map(|binding| CudaRawBinding {
                buffer: &binding.buffer.raw,
                offset_bytes: binding.offset_bytes,
                length_bytes: binding.length_bytes,
                access: match binding.access
                {
                    BufferAccess::ReadOnly => CudaRawAccess::ReadOnly,
                    BufferAccess::WriteOnly => CudaRawAccess::WriteOnly,
                    BufferAccess::ReadWrite => CudaRawAccess::ReadWrite,
                },
            })
            .collect::<Vec<_>>();

        let raw = self
            .runtime
            .launch(
                &kernel.raw,
                CudaRawLaunchConfig {
                    grid: config.grid,
                    block: config.block,
                    shared_memory_bytes: config.shared_memory_bytes,
                },
                &raw_bindings,
            )
            .map_err(ComputeError::Launch)?;

        Ok(CudaComputeEvent { raw })
    }

    fn wait(&self, event: &Self::Event) -> ComputeResult<()> {
        self.runtime
            .wait(&event.raw)
            .map_err(ComputeError::Synchronization)
    }

    fn synchronize(&self, _stream: &Self::Stream) -> ComputeResult<()> {
        self.runtime
            .synchronize()
            .map_err(ComputeError::Synchronization)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter_or_skip() -> Option<CudaComputeAdapter> {
        match CudaComputeAdapter::new()
        {
            Ok(adapter) => Some(adapter),
            Err(error) =>
            {
                eprintln!("cuda: {error}; skipping compute-adapter test");
                None
            },
        }
    }

    #[test]
    fn adapter_reports_real_cuda_capabilities() {
        let Some(adapter) = adapter_or_skip()
        else
        {
            return;
        };

        let capabilities = adapter.capabilities();

        assert_eq!(capabilities.device.kind(), DeviceKind::Cuda);
        assert!(capabilities.name.contains("scirust-gpu-cuda"));
        assert!(capabilities.supports_dtype(DType::F32));
        assert!(capabilities.supports_dtype(DType::Bf16));
        assert!(capabilities.max_buffer_bytes.is_some());
        assert!(
            capabilities
                .max_workgroup_size
                .iter()
                .all(|value| *value > 0)
        );
        assert!(capabilities.supports_async_execution);
    }

    #[test]
    fn device_buffer_round_trip_is_checked() {
        let Some(adapter) = adapter_or_skip()
        else
        {
            return;
        };

        let buffer = adapter
            .allocate(8, 1, MemorySpace::Device)
            .expect("CUDA allocation");

        adapter
            .write(&buffer, 2, &[11, 22, 33, 44])
            .expect("CUDA upload");

        let mut output = [0_u8; 4];
        adapter
            .read(&buffer, 2, &mut output)
            .expect("CUDA download");

        assert_eq!(output, [11, 22, 33, 44]);
        assert!(adapter.write(&buffer, 7, &[1, 2]).is_err());
        assert!(adapter.allocate(8, 4, MemorySpace::Device).is_err());
        assert!(adapter.allocate(8, 1, MemorySpace::Host).is_err());
    }

    #[test]
    fn ptx_kernel_executes_through_compute_contract() {
        let Some(adapter) = adapter_or_skip()
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

        let buffer = adapter
            .allocate(16, 1, MemorySpace::Device)
            .expect("CUDA allocation");

        let input = [10_u32, 20, 30, 40];
        let mut input_bytes = Vec::with_capacity(16);
        for value in input
        {
            input_bytes.extend_from_slice(&value.to_ne_bytes());
        }

        adapter
            .write(&buffer, 0, &input_bytes)
            .expect("CUDA upload");

        let module = KernelModule::new(KernelFormat::Ptx, "increment_u32", PTX.as_bytes().to_vec())
            .expect("valid PTX module");

        let kernel = adapter.compile(&module).expect("PTX loading");
        let stream = adapter.create_stream().expect("logical CUDA stream");
        let config = LaunchConfig::new([1, 1, 1], [4, 1, 1], 0).expect("valid launch");

        let bindings = [BufferBinding {
            slot: 0,
            buffer: &buffer,
            offset_bytes: 0,
            length_bytes: 16,
            access: BufferAccess::ReadWrite,
        }];

        let event = adapter
            .launch(&kernel, &stream, config, &bindings)
            .expect("CUDA launch");

        adapter.wait(&event).expect("CUDA completion");

        let mut output_bytes = [0_u8; 16];
        adapter
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
        adapter.synchronize(&stream).expect("CUDA synchronization");
    }

    #[test]
    fn duplicate_binding_slots_are_rejected() {
        let Some(adapter) = adapter_or_skip()
        else
        {
            return;
        };

        const PTX: &str = r#"
.version 8.0
.target sm_80
.address_size 64

.visible .entry no_op(
    .param .u64 no_op_param_0
)
{
    ret;
}
"#;

        let buffer = adapter
            .allocate(4, 1, MemorySpace::Device)
            .expect("CUDA allocation");

        let module = KernelModule::new(KernelFormat::Ptx, "no_op", PTX.as_bytes().to_vec())
            .expect("valid PTX module");

        let kernel = adapter.compile(&module).expect("PTX loading");
        let stream = adapter.create_stream().expect("logical CUDA stream");
        let config = LaunchConfig::new([1, 1, 1], [1, 1, 1], 0).expect("valid launch");

        let bindings = [
            BufferBinding {
                slot: 0,
                buffer: &buffer,
                offset_bytes: 0,
                length_bytes: 4,
                access: BufferAccess::ReadOnly,
            },
            BufferBinding {
                slot: 0,
                buffer: &buffer,
                offset_bytes: 0,
                length_bytes: 4,
                access: BufferAccess::ReadOnly,
            },
        ];

        assert!(matches!(
            adapter.launch(&kernel, &stream, config, &bindings),
            Err(ComputeError::InvalidArgument(
                "buffer binding slots must be unique"
            ))
        ));

        let aliased_bindings = [
            BufferBinding {
                slot: 0,
                buffer: &buffer,
                offset_bytes: 0,
                length_bytes: 4,
                access: BufferAccess::ReadOnly,
            },
            BufferBinding {
                slot: 1,
                buffer: &buffer,
                offset_bytes: 0,
                length_bytes: 4,
                access: BufferAccess::ReadOnly,
            },
        ];

        assert!(matches!(
            adapter.launch(&kernel, &stream, config, &aliased_bindings),
            Err(ComputeError::Launch(message))
                if message.contains("same allocation")
        ));
    }
}
