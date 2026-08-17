use crate::{
    BufferBinding, ComputeBackend, ComputeResult, DeviceCapabilities, HardwareCapabilities,
    KernelModule, LaunchConfig, MemorySpace,
};

/// Allocation and transfer plane of a compute backend.
pub trait BackendAllocator: ComputeBackend {
    fn allocate_buffer(
        &self,
        bytes: usize,
        alignment: usize,
        memory_space: MemorySpace,
    ) -> ComputeResult<Self::Buffer> {
        self.allocate(bytes, alignment, memory_space)
    }

    fn write_buffer(
        &self,
        destination: &Self::Buffer,
        offset_bytes: usize,
        data: &[u8],
    ) -> ComputeResult<()> {
        self.write(destination, offset_bytes, data)
    }

    fn read_buffer(
        &self,
        source: &Self::Buffer,
        offset_bytes: usize,
        destination: &mut [u8],
    ) -> ComputeResult<()> {
        self.read(source, offset_bytes, destination)
    }
}
impl<T: ComputeBackend + ?Sized> BackendAllocator for T {}

/// Kernel compilation plane of a compute backend.
pub trait BackendCompiler: ComputeBackend {
    fn compile_kernel(&self, module: &KernelModule) -> ComputeResult<Self::Kernel> {
        self.compile(module)
    }
}
impl<T: ComputeBackend + ?Sized> BackendCompiler for T {}

/// Stream/event/dispatch plane of a compute backend.
pub trait BackendExecutor: ComputeBackend {
    fn new_stream(&self) -> ComputeResult<Self::Stream> {
        self.create_stream()
    }

    fn dispatch(
        &self,
        kernel: &Self::Kernel,
        stream: &Self::Stream,
        config: LaunchConfig,
        bindings: &[BufferBinding<'_, Self::Buffer>],
    ) -> ComputeResult<Self::Event> {
        self.launch(kernel, stream, config, bindings)
    }

    fn wait_event(&self, event: &Self::Event) -> ComputeResult<()> {
        self.wait(event)
    }

    fn synchronize_stream(&self, stream: &Self::Stream) -> ComputeResult<()> {
        self.synchronize(stream)
    }
}
impl<T: ComputeBackend + ?Sized> BackendExecutor for T {}

/// Capability/introspection plane used by compilers and schedulers.
pub trait BackendIntrospection: ComputeBackend {
    fn device_capabilities(&self) -> &DeviceCapabilities {
        self.capabilities()
    }

    fn architecture_capabilities(&self) -> HardwareCapabilities {
        self.hardware_capabilities()
    }

    fn supports_dtype(&self, dtype: crate::DType) -> bool {
        self.capabilities().supports_dtype(dtype)
    }
}
impl<T: ComputeBackend + ?Sized> BackendIntrospection for T {}

/// Complete Core2 backend contract assembled from orthogonal planes.
pub trait BackendRuntime:
    ComputeBackend + BackendAllocator + BackendCompiler + BackendExecutor + BackendIntrospection
{
}

impl<T> BackendRuntime for T where
    T: ComputeBackend + BackendAllocator + BackendCompiler + BackendExecutor + BackendIntrospection
{
}
