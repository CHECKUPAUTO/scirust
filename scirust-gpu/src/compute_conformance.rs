use scirust_compute::{
    ComputeBackend, ComputeError, DeviceKind, KernelFormat, KernelModule, MemorySpace,
};

use crate::{CpuBuffer, CpuComputeAdapter};

#[cfg(feature = "cuda")]
use crate::{CudaComputeAdapter, CudaComputeBuffer};
#[cfg(feature = "wgpu")]
use crate::{WgpuComputeAdapter, WgpuComputeBuffer};

const BUFFER_BYTES: usize = 16;
const TRANSFER_OFFSET: usize = 4;
const TRANSFER_DATA: [u8; 4] = [11, 22, 33, 44];

trait ConformanceProfile {
    type Backend: ComputeBackend;

    fn adapter() -> Option<Self::Backend>;
    fn kind() -> DeviceKind;
    fn memory_space() -> MemorySpace;
    fn wrong_memory_space() -> MemorySpace;
    fn alignment() -> usize;
    fn wrong_alignment() -> usize;

    fn buffer_len(buffer: &<Self::Backend as ComputeBackend>::Buffer) -> usize;
    fn buffer_alignment(buffer: &<Self::Backend as ComputeBackend>::Buffer) -> usize;
    fn buffer_memory_space(buffer: &<Self::Backend as ComputeBackend>::Buffer) -> MemorySpace;

    fn supported_module() -> KernelModule;
    fn unsupported_module() -> KernelModule;
}

fn capabilities_conform<P: ConformanceProfile>() {
    let Some(adapter) = P::adapter()
    else
    {
        return;
    };

    let capabilities = adapter.capabilities();

    assert_eq!(capabilities.device.kind(), P::kind());
    assert!(!capabilities.name.is_empty());
    assert!(!capabilities.supported_dtypes.is_empty());
    assert!(
        capabilities
            .max_workgroup_size
            .iter()
            .all(|dimension| *dimension > 0)
    );
}

fn memory_and_transfer_contract_conforms<P: ConformanceProfile>() {
    let Some(adapter) = P::adapter()
    else
    {
        return;
    };

    assert!(matches!(
        adapter.allocate(BUFFER_BYTES, 0, P::memory_space()),
        Err(ComputeError::InvalidArgument(_))
    ));

    assert!(matches!(
        adapter.allocate(BUFFER_BYTES, P::alignment(), P::wrong_memory_space()),
        Err(ComputeError::Unsupported(_))
    ));

    assert!(matches!(
        adapter.allocate(BUFFER_BYTES, P::wrong_alignment(), P::memory_space()),
        Err(ComputeError::Unsupported(_))
    ));

    let buffer = adapter
        .allocate(BUFFER_BYTES, P::alignment(), P::memory_space())
        .expect("conforming allocation");

    assert_eq!(P::buffer_len(&buffer), BUFFER_BYTES);
    assert_eq!(P::buffer_alignment(&buffer), P::alignment());
    assert_eq!(P::buffer_memory_space(&buffer), P::memory_space());

    adapter
        .write(&buffer, TRANSFER_OFFSET, &TRANSFER_DATA)
        .expect("conforming write");

    let mut output = [0_u8; TRANSFER_DATA.len()];
    adapter
        .read(&buffer, TRANSFER_OFFSET, &mut output)
        .expect("conforming read");

    assert_eq!(output, TRANSFER_DATA);

    assert!(matches!(
        adapter.write(&buffer, BUFFER_BYTES, &TRANSFER_DATA),
        Err(ComputeError::Transfer(_))
    ));

    assert!(matches!(
        adapter.read(&buffer, BUFFER_BYTES, &mut output),
        Err(ComputeError::Transfer(_))
    ));

    assert!(matches!(
        adapter.write(&buffer, usize::MAX, &TRANSFER_DATA),
        Err(ComputeError::Transfer(_))
    ));

    assert!(matches!(
        adapter.read(&buffer, usize::MAX, &mut output),
        Err(ComputeError::Transfer(_))
    ));
}

fn kernel_and_stream_contract_conforms<P: ConformanceProfile>() {
    let Some(adapter) = P::adapter()
    else
    {
        return;
    };

    let _kernel = adapter
        .compile(&P::supported_module())
        .expect("supported kernel format");

    assert!(matches!(
        adapter.compile(&P::unsupported_module()),
        Err(ComputeError::Unsupported(_))
    ));

    let stream = adapter.create_stream().expect("stream creation");
    adapter
        .synchronize(&stream)
        .expect("stream synchronization");
}

struct CpuProfile;

impl ConformanceProfile for CpuProfile {
    type Backend = CpuComputeAdapter;

    fn adapter() -> Option<Self::Backend> {
        Some(CpuComputeAdapter::new())
    }

    fn kind() -> DeviceKind {
        DeviceKind::Cpu
    }

    fn memory_space() -> MemorySpace {
        MemorySpace::Host
    }

    fn wrong_memory_space() -> MemorySpace {
        MemorySpace::Device
    }

    fn alignment() -> usize {
        1
    }

    fn wrong_alignment() -> usize {
        4
    }

    fn buffer_len(buffer: &CpuBuffer) -> usize {
        buffer.len()
    }

    fn buffer_alignment(buffer: &CpuBuffer) -> usize {
        buffer.alignment()
    }

    fn buffer_memory_space(buffer: &CpuBuffer) -> MemorySpace {
        buffer.memory_space()
    }

    /// A genuine Reference artefact.
    ///
    /// This used to be nine arbitrary bytes (`b"reference"`), which passed only
    /// because `compile` did not parse its payload. Now that the CPU adapter
    /// decodes and validates Reference modules, the conformance suite has to
    /// supply a real one.
    fn supported_module() -> KernelModule {
        crate::cpu_reference::test_support::canonical_reference_module()
    }

    fn unsupported_module() -> KernelModule {
        KernelModule::new(KernelFormat::Wgsl, "main", b"unsupported".to_vec())
            .expect("valid unsupported module")
    }
}

#[cfg(feature = "wgpu")]
struct WgpuProfile;

#[cfg(feature = "wgpu")]
impl ConformanceProfile for WgpuProfile {
    type Backend = WgpuComputeAdapter;

    fn adapter() -> Option<Self::Backend> {
        WgpuComputeAdapter::new().ok()
    }

    fn kind() -> DeviceKind {
        DeviceKind::Wgpu
    }

    fn memory_space() -> MemorySpace {
        MemorySpace::Device
    }

    fn wrong_memory_space() -> MemorySpace {
        MemorySpace::Host
    }

    fn alignment() -> usize {
        4
    }

    fn wrong_alignment() -> usize {
        1
    }

    fn buffer_len(buffer: &WgpuComputeBuffer) -> usize {
        buffer.len()
    }

    fn buffer_alignment(buffer: &WgpuComputeBuffer) -> usize {
        buffer.alignment()
    }

    fn buffer_memory_space(buffer: &WgpuComputeBuffer) -> MemorySpace {
        buffer.memory_space()
    }

    fn supported_module() -> KernelModule {
        const WGSL: &str = r#"
@compute @workgroup_size(1)
fn main() {}
"#;

        KernelModule::new(KernelFormat::Wgsl, "main", WGSL.as_bytes().to_vec())
            .expect("valid WGSL module")
    }

    fn unsupported_module() -> KernelModule {
        KernelModule::new(KernelFormat::Reference, "main", b"unsupported".to_vec())
            .expect("valid unsupported module")
    }
}

#[cfg(feature = "cuda")]
struct CudaProfile;

#[cfg(feature = "cuda")]
impl ConformanceProfile for CudaProfile {
    type Backend = CudaComputeAdapter;

    fn adapter() -> Option<Self::Backend> {
        CudaComputeAdapter::new().ok()
    }

    fn kind() -> DeviceKind {
        DeviceKind::Cuda
    }

    fn memory_space() -> MemorySpace {
        MemorySpace::Device
    }

    fn wrong_memory_space() -> MemorySpace {
        MemorySpace::Host
    }

    fn alignment() -> usize {
        1
    }

    fn wrong_alignment() -> usize {
        4
    }

    fn buffer_len(buffer: &CudaComputeBuffer) -> usize {
        buffer.len()
    }

    fn buffer_alignment(buffer: &CudaComputeBuffer) -> usize {
        buffer.alignment()
    }

    fn buffer_memory_space(buffer: &CudaComputeBuffer) -> MemorySpace {
        buffer.memory_space()
    }

    fn supported_module() -> KernelModule {
        const PTX: &str = r#"
.version 8.0
.target sm_80
.address_size 64

.visible .entry main()
{
    ret;
}
"#;

        KernelModule::new(KernelFormat::Ptx, "main", PTX.as_bytes().to_vec())
            .expect("valid PTX module")
    }

    fn unsupported_module() -> KernelModule {
        KernelModule::new(KernelFormat::Reference, "main", b"unsupported".to_vec())
            .expect("valid unsupported module")
    }
}

macro_rules! conformance_tests {
    ($profile:ty, $capabilities:ident, $memory:ident, $kernel:ident) => {
        #[test]
        fn $capabilities() {
            capabilities_conform::<$profile>();
        }

        #[test]
        fn $memory() {
            memory_and_transfer_contract_conforms::<$profile>();
        }

        #[test]
        fn $kernel() {
            kernel_and_stream_contract_conforms::<$profile>();
        }
    };
}

conformance_tests!(
    CpuProfile,
    cpu_capabilities_conform,
    cpu_memory_and_transfer_contract_conforms,
    cpu_kernel_and_stream_contract_conforms
);

#[cfg(feature = "wgpu")]
conformance_tests!(
    WgpuProfile,
    wgpu_capabilities_conform,
    wgpu_memory_and_transfer_contract_conforms,
    wgpu_kernel_and_stream_contract_conforms
);

#[cfg(feature = "cuda")]
conformance_tests!(
    CudaProfile,
    cuda_capabilities_conform,
    cuda_memory_and_transfer_contract_conforms,
    cuda_kernel_and_stream_contract_conforms
);
