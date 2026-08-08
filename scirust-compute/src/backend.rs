use crate::{
    BufferBinding, ComputeResult, DeviceCapabilities, ExecutionLimits, HardwareCapabilities,
    KernelModule, LaunchConfig, MemorySpace,
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
    /// The default starts from the conservative legacy bridge. With `std`
    /// enabled, a CPU backend additionally receives the runtime host
    /// architecture/ISA facts from [`crate::probe_host_cpu`]. Numeric, memory,
    /// matrix and reproducibility guarantees are still left to concrete
    /// backends to override when they can state them honestly.
    fn hardware_capabilities(&self) -> HardwareCapabilities {
        let mut hardware = self.capabilities().hardware_baseline();

        #[cfg(feature = "std")]
        if self.capabilities().device.kind() == crate::DeviceKind::Cpu
        {
            let probed = crate::probe_host_cpu();
            hardware.architecture = probed.architecture;
            hardware.isa = probed.isa;
        }

        hardware
    }

    /// Portable execution limits known by this backend.
    ///
    /// The default preserves the launch-width facts already carried by the
    /// legacy [`DeviceCapabilities`] contract. Backends may override this as the
    /// generic limit model grows richer; callers should use this method rather
    /// than reconstructing limits from backend-specific state.
    fn execution_limits(&self) -> ExecutionLimits {
        ExecutionLimits::from_device_capabilities(self.capabilities())
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
