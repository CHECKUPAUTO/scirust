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
    ///
    /// This is an identity observation only. It does not claim any optional ISA
    /// feature such as AVX-512, SVE or RVV; those require explicit probing.
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
///
/// Known variants cover the first portability targets. `Other` is an escape
/// hatch for future ISAs and vendor extensions without changing this contract.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum IsaFeature {
    // x86-64
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
    // AArch64
    Neon,
    DotProd,
    I8mm,
    ArmBf16,
    Sve,
    Sve2,
    Sme,
    Sme2,
    // RISC-V and LoongArch
    RiscVVector,
    LoongArchLsx,
    LoongArchLasx,
    // Future or vendor-defined feature.
    Other(String),
}

/// Shape of the vector execution model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum VectorModel {
    Scalar,
    FixedWidth,
    Scalable,
    #[default]
    Unknown,
}

/// Instruction-set capabilities known for one processor.
///
/// `features` contains capabilities positively detected as supported;
/// `unsupported_features` contains capabilities for which a reliable probe ran
/// and returned false. A feature in neither list remains [`SupportLevel::Unknown`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IsaCapabilities {
    pub features: Vec<IsaFeature>,
    pub unsupported_features: Vec<IsaFeature>,
    pub vector_model: VectorModel,
    /// Minimum vector width in bits when the backend can state one.
    pub min_vector_bits: Option<u32>,
    /// Maximum vector width in bits when fixed or bounded and known.
    pub max_vector_bits: Option<u32>,
}

impl IsaCapabilities {
    pub fn support_level(&self, feature: &IsaFeature) -> SupportLevel {
        if self.features.contains(feature)
        {
            SupportLevel::Supported
        }
        else if self.unsupported_features.contains(feature)
        {
            SupportLevel::Unsupported
        }
        else
        {
            SupportLevel::Unknown
        }
    }

    pub fn supports(&self, feature: &IsaFeature) -> bool {
        self.support_level(feature).is_supported()
    }
}

/// Numeric formats explicitly advertised by a backend.
///
/// An empty list means "not advertised/unknown", not "unsupported". Storage,
/// arithmetic and accumulation are separate because accelerators frequently
/// store one format while accumulating in another.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NumericCapabilities {
    pub storage_dtypes: Vec<DType>,
    pub arithmetic_dtypes: Vec<DType>,
    pub accumulation_dtypes: Vec<DType>,
}

/// Matrix/tensor acceleration properties.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MatrixCapabilities {
    pub accelerated: SupportLevel,
    pub input_dtypes: Vec<DType>,
    pub accumulation_dtypes: Vec<DType>,
}

/// Memory properties relevant to portable scheduling and transfer planning.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MemoryCapabilities {
    /// Memory spaces explicitly accepted by this backend when known.
    pub spaces: Vec<MemorySpace>,
    pub coherent_host_device: SupportLevel,
    pub unified_addressing: SupportLevel,
    pub async_transfers: SupportLevel,
}

/// Execution properties independent of a particular kernel language.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ExecutionCapabilities {
    pub async_execution: SupportLevel,
    pub ordered_streams: SupportLevel,
    pub subgroup_operations: SupportLevel,
    pub atomic_i64: SupportLevel,
}

/// Reproducibility mode a backend/kernel combination can promise.
///
/// These are semantic categories, not a numeric ordering. A backend may expose
/// several modes depending on the selected kernel implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ReproducibilityLevel {
    /// Same output bits for the same inputs under the declared contract.
    BitExact,
    /// Stable result on the same backend/implementation, without cross-backend
    /// bit identity being implied.
    Deterministic,
    /// Numerically equivalent within a separately declared tolerance contract.
    NumericallyEquivalent,
    /// Performance-oriented mode that permits documented approximation.
    FastApproximate,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReproducibilityCapabilities {
    pub modes: Vec<ReproducibilityLevel>,
}

impl ReproducibilityCapabilities {
    pub fn supports(&self, level: ReproducibilityLevel) -> bool {
        self.modes.contains(&level)
    }
}

/// Architecture-neutral capability profile for one logical compute device.
///
/// The profile is intentionally additive to [`DeviceCapabilities`]. Existing
/// backends can continue exposing the legacy resource limits while gradually
/// overriding the richer hardware profile as reliable probes become available.
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
    ///
    /// Only facts already represented by [`DeviceCapabilities`] are promoted.
    /// Everything else remains unknown until a backend or hardware probe states
    /// it explicitly.
    pub fn from_device_capabilities(capabilities: &DeviceCapabilities) -> Self {
        let architecture = match capabilities.device.kind()
        {
            DeviceKind::Reference | DeviceKind::Cpu => Architecture::current_host(),
            _ => Architecture::unknown(),
        };

        Self {
            device: capabilities.device,
            architecture,
            isa: IsaCapabilities::default(),
            numeric: NumericCapabilities {
                storage_dtypes: capabilities.supported_dtypes.clone(),
                arithmetic_dtypes: Vec::new(),
                accumulation_dtypes: Vec::new(),
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
    fn current_host_architecture_never_claims_optional_isa_features() {
        let capabilities =
            HardwareCapabilities::from_device_capabilities(&DeviceCapabilities::reference_cpu());

        assert_eq!(capabilities.device, DeviceId::reference());
        assert!(capabilities.isa.features.is_empty());
        assert!(capabilities.isa.unsupported_features.is_empty());
        assert_eq!(capabilities.isa.vector_model, VectorModel::Unknown);
    }

    #[test]
    fn legacy_bridge_promotes_only_known_facts() {
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
            hardware.numeric.storage_dtypes,
            vec![DType::F16, DType::F32]
        );
        assert!(hardware.numeric.arithmetic_dtypes.is_empty());
        assert_eq!(hardware.execution.async_execution, SupportLevel::Supported);
        assert_eq!(hardware.matrix.accelerated, SupportLevel::Unknown);
        assert!(hardware.memory.spaces.is_empty());
    }

    #[test]
    fn feature_support_preserves_supported_unsupported_and_unknown() {
        let isa = IsaCapabilities {
            features: vec![IsaFeature::Avx2, IsaFeature::Fma],
            unsupported_features: vec![IsaFeature::Avx512F],
            ..IsaCapabilities::default()
        };

        assert_eq!(isa.support_level(&IsaFeature::Avx2), SupportLevel::Supported);
        assert_eq!(
            isa.support_level(&IsaFeature::Avx512F),
            SupportLevel::Unsupported
        );
        assert_eq!(isa.support_level(&IsaFeature::Sse2), SupportLevel::Unknown);
        assert!(isa.supports(&IsaFeature::Avx2));
        assert!(!isa.supports(&IsaFeature::Avx512F));
    }

    #[test]
    fn reproducibility_membership_is_explicit() {
        let reproducibility = ReproducibilityCapabilities {
            modes: vec![ReproducibilityLevel::Deterministic],
        };

        assert!(reproducibility.supports(ReproducibilityLevel::Deterministic));
        assert!(!reproducibility.supports(ReproducibilityLevel::BitExact));
    }
}
