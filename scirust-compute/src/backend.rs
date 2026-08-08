use crate::{
    BufferBinding, ComputeResult, DeviceCapabilities, HardwareCapabilities, KernelModule,
    LaunchConfig, MemorySpace,
};

/// Backend-neutral execution contract.
///
/// Graph compilation, autograd, scheduling and memory policies remain outside
/// this trait.
pub trait ComputeBackend {
    type Buffer;
    type Kernel;
    type Stream;
    type Event;

    fn capabilities(&self) -> &DeviceCapabilities;

    /// Rich architecture-neutral hardware profile for this backend.
    ///
    /// The default implementation is deliberately conservative and promotes
    /// only facts already present in [`DeviceCapabilities`]. Backends should
    /// override this method when they have reliable ISA, numeric, memory,
    /// matrix or reproducibility information.
    fn hardware_capabilities(&self) -> HardwareCapabilities {
        self.capabilities().hardware_baseline()
    }

    fn allocate(
        &self,
        bytes: usize,
        alignment: usize,
        memory_space: MemorySpace,
    ) -> ComputeResult<Self::Buffer>;

    fn write(
        &self,
        destination: &Self::Buffer,
        offset_bytes: usize,
        data: &[u8],
    ) -> ComputeResult<()>;

    fn read(
        &self,
        source: &Self::Buffer,
        offset_bytes: usize,
        destination: &mut [u8],
    ) -> ComputeResult<()>;

    fn compile(&self, module: &KernelModule) -> ComputeResult<Self::Kernel>;

    fn create_stream(&self) -> ComputeResult<Self::Stream>;

    fn launch(
        &self,
        kernel: &Self::Kernel,
        stream: &Self::Stream,
        config: LaunchConfig,
        bindings: &[BufferBinding<'_, Self::Buffer>],
    ) -> ComputeResult<Self::Event>;

    fn wait(&self, event: &Self::Event) -> ComputeResult<()>;

    fn synchronize(&self, stream: &Self::Stream) -> ComputeResult<()>;
}