extern crate alloc;

use alloc::{string::ToString, vec, vec::Vec};
use core::cell::RefCell;

use crate::{BackendResult, CpuBackend, RawComputeBackend};

use scirust_compute::{
    BufferBinding, ComputeBackend, ComputeError, ComputeResult, DType, DeviceCapabilities,
    DeviceId, KernelFormat, KernelModule, LaunchConfig, MemorySpace,
};

/// Adapter exposing the deterministic SciRust CPU path through
/// `scirust_compute::ComputeBackend`.
#[derive(Debug)]
pub struct CpuComputeAdapter {
    capabilities: DeviceCapabilities,
}

impl CpuComputeAdapter {
    pub fn new() -> Self {
        Self {
            capabilities: DeviceCapabilities {
                device: DeviceId::cpu(),
                name: "scirust-gpu-cpu".to_string(),
                supported_dtypes: vec![DType::F32],
                max_buffer_bytes: None,
                max_workgroup_size: [1, 1, 1],
                supports_async_execution: false,
            },
        }
    }

    pub fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }
}

impl Default for CpuComputeAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct CpuBuffer {
    pub(crate) bytes: RefCell<Vec<u8>>,
    alignment: usize,
    memory_space: MemorySpace,
}

impl CpuBuffer {
    pub fn len(&self) -> usize {
        self.bytes.borrow().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn alignment(&self) -> usize {
        self.alignment
    }

    pub fn memory_space(&self) -> MemorySpace {
        self.memory_space
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuKernel {
    pub(crate) module: KernelModule,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuStream(());

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuEvent(());

impl RawComputeBackend for CpuComputeAdapter {
    fn device_name(&self) -> &'static str {
        CpuBackend.device_name()
    }

    fn gemm_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> BackendResult<Vec<f32>> {
        CpuBackend.gemm_f32(a, b, m, k, n)
    }
}

impl ComputeBackend for CpuComputeAdapter {
    type Buffer = CpuBuffer;
    type Kernel = CpuKernel;
    type Stream = CpuStream;
    type Event = CpuEvent;

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
        if alignment != 1
        {
            return Err(ComputeError::Unsupported(
                "CPU adapter currently supports byte alignment only",
            ));
        }
        if memory_space != MemorySpace::Host
        {
            return Err(ComputeError::Unsupported(
                "CPU adapter supports host memory only",
            ));
        }

        let mut storage = Vec::new();
        storage
            .try_reserve_exact(bytes)
            .map_err(|error| ComputeError::Allocation(error.to_string()))?;
        storage.resize(bytes, 0);

        Ok(CpuBuffer {
            bytes: RefCell::new(storage),
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
        let end = offset_bytes
            .checked_add(data.len())
            .ok_or_else(|| ComputeError::Transfer("write range overflow".to_string()))?;

        let mut bytes = destination
            .bytes
            .try_borrow_mut()
            .map_err(|_| ComputeError::Transfer("buffer is already borrowed".to_string()))?;

        if end > bytes.len()
        {
            return Err(ComputeError::Transfer(
                "write exceeds buffer bounds".to_string(),
            ));
        }

        bytes[offset_bytes..end].copy_from_slice(data);
        Ok(())
    }

    fn read(
        &self,
        source: &Self::Buffer,
        offset_bytes: usize,
        destination: &mut [u8],
    ) -> ComputeResult<()> {
        let end = offset_bytes
            .checked_add(destination.len())
            .ok_or_else(|| ComputeError::Transfer("read range overflow".to_string()))?;

        let bytes = source
            .bytes
            .try_borrow()
            .map_err(|_| ComputeError::Transfer("buffer is mutably borrowed".to_string()))?;

        if end > bytes.len()
        {
            return Err(ComputeError::Transfer(
                "read exceeds buffer bounds".to_string(),
            ));
        }

        destination.copy_from_slice(&bytes[offset_bytes..end]);
        Ok(())
    }

    fn compile(&self, module: &KernelModule) -> ComputeResult<Self::Kernel> {
        if module.format != KernelFormat::Reference
        {
            return Err(ComputeError::Unsupported(
                "CPU adapter accepts reference kernels only",
            ));
        }

        Ok(CpuKernel {
            module: module.clone(),
        })
    }

    fn create_stream(&self) -> ComputeResult<Self::Stream> {
        Ok(CpuStream(()))
    }

    fn launch(
        &self,
        _kernel: &Self::Kernel,
        _stream: &Self::Stream,
        _config: LaunchConfig,
        _bindings: &[BufferBinding<'_, Self::Buffer>],
    ) -> ComputeResult<Self::Event> {
        Err(ComputeError::Unsupported(
            "CPU reference kernel launch ABI is not implemented",
        ))
    }

    fn wait(&self, _event: &Self::Event) -> ComputeResult<()> {
        Ok(())
    }

    fn synchronize(&self, _stream: &Self::Stream) -> ComputeResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_reports_honest_cpu_capabilities() {
        let adapter = CpuComputeAdapter::new();

        assert_eq!(adapter.capabilities().device, DeviceId::cpu());
        assert!(adapter.capabilities().supports_dtype(DType::F32));
        assert!(!adapter.capabilities().supports_async_execution);
    }

    #[test]
    fn adapter_preserves_the_existing_cpu_gemm_contract() {
        let adapter = CpuComputeAdapter::new();
        let a = [1.0, 2.0, 3.0, 4.0];
        let identity = [1.0, 0.0, 0.0, 1.0];

        assert_eq!(adapter.device_name(), "cpu");
        assert_eq!(
            adapter.gemm_f32(&a, &identity, 2, 2, 2).unwrap(),
            a.to_vec()
        );
    }

    #[test]
    fn host_buffer_round_trip_is_checked() {
        let adapter = CpuComputeAdapter::new();
        let buffer = adapter.allocate(8, 1, MemorySpace::Host).unwrap();

        assert_eq!(buffer.len(), 8);
        assert_eq!(buffer.alignment(), 1);
        assert_eq!(buffer.memory_space(), MemorySpace::Host);

        adapter.write(&buffer, 2, &[10, 20, 30]).unwrap();

        let mut output = [0u8; 8];
        adapter.read(&buffer, 0, &mut output).unwrap();

        assert_eq!(output, [0, 0, 10, 20, 30, 0, 0, 0]);
    }

    #[test]
    fn invalid_buffer_requests_are_rejected() {
        let adapter = CpuComputeAdapter::new();

        assert!(matches!(
            adapter.allocate(8, 0, MemorySpace::Host),
            Err(ComputeError::InvalidArgument(_))
        ));

        assert!(matches!(
            adapter.allocate(8, 1, MemorySpace::Device),
            Err(ComputeError::Unsupported(_))
        ));

        let buffer = adapter.allocate(4, 1, MemorySpace::Host).unwrap();

        assert!(matches!(
            adapter.write(&buffer, 3, &[1, 2]),
            Err(ComputeError::Transfer(_))
        ));
    }

    #[test]
    fn reference_kernel_compilation_is_explicit() {
        let adapter = CpuComputeAdapter::new();

        let reference =
            KernelModule::new(KernelFormat::Reference, "main", b"reference".to_vec()).unwrap();
        let kernel = adapter.compile(&reference).unwrap();

        assert_eq!(kernel.module, reference);

        let wgsl = KernelModule::new(KernelFormat::Wgsl, "main", b"wgsl".to_vec()).unwrap();

        assert!(matches!(
            adapter.compile(&wgsl),
            Err(ComputeError::Unsupported(_))
        ));
    }

    #[test]
    fn execution_contract_is_synchronous_and_honest() {
        let adapter = CpuComputeAdapter::new();
        let stream = adapter.create_stream().unwrap();
        let kernel = adapter
            .compile(
                &KernelModule::new(KernelFormat::Reference, "main", b"reference".to_vec()).unwrap(),
            )
            .unwrap();
        let config = LaunchConfig::new([1, 1, 1], [1, 1, 1], 0).unwrap();

        assert!(matches!(
            adapter.launch(&kernel, &stream, config, &[]),
            Err(ComputeError::Unsupported(_))
        ));

        adapter.wait(&CpuEvent(())).unwrap();
        adapter.synchronize(&stream).unwrap();
    }
}
