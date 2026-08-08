extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use crate::{DType, DeviceCapabilities, DeviceId, DeviceKind, MemorySpace};

/// Tri-state support declaration for hardware properties that may be unknown.
///
/// `Unknown` is intentionally distinct from `Unsupported`: probing code and
/// backend adapters must not turn a missing observation into a negative claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum SupportLevel {
    #[default]
    Unknown,
    Unsupported,
    Supported,
}

impl SupportLevel {
    pub const fn is_supported(self) -> bool {
        matches!(self, Self::Supported)
    }

    pub const fn from_known(value: bool) -> Self {
        if value
        {
            Self::Supported
        }
        else
        {
            Self::Unsupported
        }
    }
}

/// A finite set of values with explicit supported/unsupported/unknown state.
///
/// Values absent from both partitions are unknown. Mutating through
/// [`CapabilitySet::set_support`] keeps the supported and unsupported partitions
/// disjoint, so a profile never needs a sentinel value to represent missing
/// knowledge.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CapabilitySet<T> {
    supported: Vec<T>,
    unsupported: Vec<T>,
}

impl<T: PartialEq> CapabilitySet<T> {
    pub fn support_level(&self, value: &T) -> SupportLevel {
        if self.supported.contains(value)
        {
            SupportLevel::Supported
        }
        else if self.unsupported.contains(value)
        {
            SupportLevel::Unsupported
        }
        else
        {
            SupportLevel::Unknown
        }
    }

    pub fn supports(&self, value: &T) -> bool {
        self.support_level(value).is_supported()
    }

    pub fn supported_values(&self) -> &[T] {
        &self.supported
    }

    pub fn unsupported_values(&self) -> &[T] {
        &self.unsupported
    }

    pub fn set_support(&mut self, value: T, level: SupportLevel) {
        self.supported.retain(|candidate| candidate != &value);
        self.unsupported.retain(|candidate| candidate != &value);

        match level
        {
            SupportLevel::Supported => self.supported.push(value),
            SupportLevel::Unsupported => self.unsupported.push(value),
            SupportLevel::Unknown =>
            {},
        }
    }
}

/// Architecture family of the processor executing a compute backend.
///
/// This enum deliberately describes families rather than individual CPUs or
/// accelerator SKUs. `Other` plus an optional architecture name keeps the
/// contract extensible without requiring SciRust to know every future
/// architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum ArchitectureFamily {
    X86_64,
    Aarch64,
    RiscV64,
    LoongArch64,
    Wasm32,
    NvidiaGpu,
    AmdGpu,
    IntelGpu,
    AppleGpu,
    Other,
    #[default]
    Unknown,
}

/// Architecture identity advertised by a backend.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Architecture {
    pub family: ArchitectureFamily,
    /// Optional backend- or vendor-provided architecture name.
    ///
    /// Examples include a microarchitecture, GPU ISA generation or a future
    /// architecture family not yet represented by [`ArchitectureFamily`].
    pub name: Option<String>,
}

impl Architecture {
    pub const fn unknown() -> Self {
        Self {
            family: ArchitectureFamily::Unknown,
            name: None,
        }
    }

    /// Compile-time architecture of the host running this binary.
    pub fn current_host() -> Self {
        Self {
            family: CURRENT_HOST_ARCHITECTURE,
            name: None,
        }
    }

    pub fn named(family: ArchitectureFamily, name: impl Into<String>) -> Self {
        Self {
            family,
            name: Some(name.into()),
        }
    }
}

#[cfg(target_arch = "x86_64")]
const CURRENT_HOST_ARCHITECTURE: ArchitectureFamily = ArchitectureFamily::X86_64;
#[cfg(target_arch = "aarch64")]
const CURRENT_HOST_ARCHITECTURE: ArchitectureFamily = ArchitectureFamily::Aarch64;
#[cfg(target_arch = "riscv64")]
const CURRENT_HOST_ARCHITECTURE: ArchitectureFamily = ArchitectureFamily::RiscV64;
#[cfg(target_arch = "loongarch64")]
const CURRENT_HOST_ARCHITECTURE: ArchitectureFamily = ArchitectureFamily::LoongArch64;
#[cfg(target_arch = "wasm32")]
const CURRENT_HOST_ARCHITECTURE: ArchitectureFamily = ArchitectureFamily::Wasm32;
#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "riscv64",
    target_arch = "loongarch64",
    target_arch = "wasm32"
)))]
const CURRENT_HOST_ARCHITECTURE: ArchitectureFamily = ArchitectureFamily::Unknown;

/// Optional ISA feature exposed by a compute processor.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum IsaFeature {
    Sse2,
    Avx2,
    Fma,
    Avx512F,
    Avx512Vnni,
    Avx512Bf16,
    Avx512Fp16,
    AmxTile,
    AmxInt8,
    AmxBf16,
    Neon,
    DotProd,
    I8mm,
    ArmBf16,
    Sve,
    Sve2,
    Sme,
    Sme2,
    RiscVVector,
    LoongArchLsx,
    LoongArchLasx,
    Other(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum VectorModel {
    Scalar,
    FixedWidth,
    Scalable,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IsaCapabilities {
    pub features: CapabilitySet<IsaFeature>,
    pub vector_model: VectorModel,
    pub min_vector_bits: Option<u32>,
    pub max_vector_bits: Option<u32>,
}

impl IsaCapabilities {
    pub fn support_level(&self, feature: &IsaFeature) -> SupportLevel {
        self.features.support_level(feature)
    }

    pub fn supports(&self, feature: &IsaFeature) -> bool {
        self.features.supports(feature)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NumericCapabilities {
    pub storage_dtypes: CapabilitySet<DType>,
    pub arithmetic_dtypes: CapabilitySet<DType>,
    pub accumulation_dtypes: CapabilitySet<DType>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MatrixCapabilities {
    pub accelerated: SupportLevel,
    pub input_dtypes: CapabilitySet<DType>,
    pub accumulation_dtypes: CapabilitySet<DType>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MemoryCapabilities {
    pub spaces: CapabilitySet<MemorySpace>,
    pub coherent_host_device: SupportLevel,
    pub unified_addressing: SupportLevel,
    pub async_transfers: SupportLevel,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ExecutionCapabilities {
    pub async_execution: SupportLevel,
    pub ordered_streams: SupportLevel,
    pub subgroup_operations: SupportLevel,
    pub atomic_i64: SupportLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ReproducibilityLevel {
    BitExact,
    Deterministic,
    NumericallyEquivalent,
    FastApproximate,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReproducibilityCapabilities {
    pub modes: CapabilitySet<ReproducibilityLevel>,
}

impl ReproducibilityCapabilities {
    pub fn support_level(&self, level: ReproducibilityLevel) -> SupportLevel {
        self.modes.support_level(&level)
    }

    pub fn supports(&self, level: ReproducibilityLevel) -> bool {
        self.modes.supports(&level)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareCapabilities {
    pub device: DeviceId,
    pub architecture: Architecture,
    pub isa: IsaCapabilities,
    pub numeric: NumericCapabilities,
    pub matrix: MatrixCapabilities,
    pub memory: MemoryCapabilities,
    pub execution: ExecutionCapabilities,
    pub reproducibility: ReproducibilityCapabilities,
}

impl HardwareCapabilities {
    /// Conservative bridge from the existing capability contract.
    pub fn from_device_capabilities(capabilities: &DeviceCapabilities) -> Self {
        let architecture = match capabilities.device.kind()
        {
            DeviceKind::Reference | DeviceKind::Cpu => Architecture::current_host(),
            _ => Architecture::unknown(),
        };

        let mut storage_dtypes = CapabilitySet::default();
        for dtype in KNOWN_DTYPES
        {
            storage_dtypes.set_support(
                dtype,
                SupportLevel::from_known(capabilities.supports_dtype(dtype)),
            );
        }

        Self {
            device: capabilities.device,
            architecture,
            isa: IsaCapabilities::default(),
            numeric: NumericCapabilities {
                storage_dtypes,
                arithmetic_dtypes: CapabilitySet::default(),
                accumulation_dtypes: CapabilitySet::default(),
            },
            matrix: MatrixCapabilities::default(),
            memory: MemoryCapabilities::default(),
            execution: ExecutionCapabilities {
                async_execution: SupportLevel::from_known(capabilities.supports_async_execution),
                ..ExecutionCapabilities::default()
            },
            reproducibility: ReproducibilityCapabilities::default(),
        }
    }
}

const KNOWN_DTYPES: [DType; 13] = [
    DType::Bool,
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
];

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn support_level_preserves_unknown_state() {
        assert!(!SupportLevel::Unknown.is_supported());
        assert_eq!(SupportLevel::from_known(true), SupportLevel::Supported);
        assert_eq!(SupportLevel::from_known(false), SupportLevel::Unsupported);
    }

    #[test]
    fn capability_set_preserves_all_three_states_and_disjointness() {
        let mut set = CapabilitySet::default();
        assert_eq!(set.support_level(&DType::F32), SupportLevel::Unknown);
        set.set_support(DType::F32, SupportLevel::Supported);
        assert_eq!(set.support_level(&DType::F32), SupportLevel::Supported);
        assert_eq!(set.supported_values(), &[DType::F32]);
        assert!(set.unsupported_values().is_empty());
        set.set_support(DType::F32, SupportLevel::Unsupported);
        assert_eq!(set.support_level(&DType::F32), SupportLevel::Unsupported);
        assert!(set.supported_values().is_empty());
        assert_eq!(set.unsupported_values(), &[DType::F32]);
        set.set_support(DType::F32, SupportLevel::Unknown);
        assert_eq!(set.support_level(&DType::F32), SupportLevel::Unknown);
    }

    #[test]
    fn current_host_architecture_never_claims_optional_isa_features() {
        let capabilities =
            HardwareCapabilities::from_device_capabilities(&DeviceCapabilities::reference_cpu());
        assert_eq!(capabilities.device, DeviceId::reference());
        assert!(capabilities.isa.features.supported_values().is_empty());
        assert!(capabilities.isa.features.unsupported_values().is_empty());
        assert_eq!(capabilities.isa.vector_model, VectorModel::Unknown);
    }

    #[test]
    fn legacy_bridge_preserves_known_storage_dtype_support() {
        let legacy = DeviceCapabilities {
            device: DeviceId::new(DeviceKind::Cuda, 2),
            name: "test-cuda".into(),
            supported_dtypes: vec![DType::F16, DType::F32],
            max_buffer_bytes: Some(1024),
            max_workgroup_size: [256, 1, 1],
            supports_async_execution: true,
        };
        let hardware = HardwareCapabilities::from_device_capabilities(&legacy);
        assert_eq!(hardware.device, legacy.device);
        assert_eq!(hardware.architecture, Architecture::unknown());
        assert_eq!(
            hardware.numeric.storage_dtypes.support_level(&DType::F16),
            SupportLevel::Supported
        );
        assert_eq!(
            hardware.numeric.storage_dtypes.support_level(&DType::F64),
            SupportLevel::Unsupported
        );
        assert_eq!(
            hardware
                .numeric
                .arithmetic_dtypes
                .support_level(&DType::F32),
            SupportLevel::Unknown
        );
        assert_eq!(hardware.execution.async_execution, SupportLevel::Supported);
        assert!(hardware.memory.spaces.supported_values().is_empty());
    }

    #[test]
    fn feature_support_preserves_supported_unsupported_and_unknown() {
        let mut isa = IsaCapabilities::default();
        isa.features
            .set_support(IsaFeature::Avx2, SupportLevel::Supported);
        isa.features
            .set_support(IsaFeature::Fma, SupportLevel::Supported);
        isa.features
            .set_support(IsaFeature::Avx512F, SupportLevel::Unsupported);
        assert_eq!(
            isa.support_level(&IsaFeature::Avx2),
            SupportLevel::Supported
        );
        assert_eq!(
            isa.support_level(&IsaFeature::Avx512F),
            SupportLevel::Unsupported
        );
        assert_eq!(isa.support_level(&IsaFeature::Sse2), SupportLevel::Unknown);
    }

    #[test]
    fn reproducibility_membership_is_explicit() {
        let mut reproducibility = ReproducibilityCapabilities::default();
        reproducibility
            .modes
            .set_support(ReproducibilityLevel::Deterministic, SupportLevel::Supported);
        assert!(reproducibility.supports(ReproducibilityLevel::Deterministic));
        assert_eq!(
            reproducibility.support_level(ReproducibilityLevel::BitExact),
            SupportLevel::Unknown
        );
    }
}
