//! Safe runtime CPU capability probing.
//!
//! Probing is a host/runtime concern and is therefore compiled only with the
//! `std` feature. The backend-neutral capability vocabulary remains available in
//! `no_std` builds.

use crate::{
    Architecture, DeviceId, ExecutionCapabilities, HardwareCapabilities, IsaCapabilities,
    IsaFeature, MatrixCapabilities, MemoryCapabilities, NumericCapabilities,
    ReproducibilityCapabilities, SupportLevel, VectorModel,
};

/// Probe the CPU executing the current process.
///
/// The returned profile deliberately contains only architecture/ISA facts.
/// Numeric backend support, memory behavior, execution queues and reproducibility
/// are properties of a concrete compute backend and therefore remain unknown.
pub fn probe_host_cpu() -> HardwareCapabilities {
    HardwareCapabilities {
        device: DeviceId::cpu(),
        architecture: Architecture::current_host(),
        isa: probe_isa(),
        numeric: NumericCapabilities::default(),
        matrix: MatrixCapabilities::default(),
        memory: MemoryCapabilities::default(),
        execution: ExecutionCapabilities::default(),
        reproducibility: ReproducibilityCapabilities::default(),
    }
}

fn record(isa: &mut IsaCapabilities, feature: IsaFeature, supported: bool) {
    isa.features
        .set_support(feature, SupportLevel::from_known(supported));
}

fn probe_isa() -> IsaCapabilities {
    #[cfg(target_arch = "x86_64")]
    {
        return probe_x86_64();
    }

    #[cfg(target_arch = "aarch64")]
    {
        return probe_aarch64();
    }

    #[cfg(target_arch = "loongarch64")]
    {
        return probe_loongarch64();
    }

    // RISC-V is intentionally conservative in this MSRV-1.89 phase. Rust can
    // identify the architecture, but SciRust does not yet rely on a stable
    // runtime RVV probe here. Leaving the ISA unknown is preferable to inferring
    // support from compile-time target features.
    #[allow(unreachable_code)]
    IsaCapabilities::default()
}

#[cfg(target_arch = "x86_64")]
fn probe_x86_64() -> IsaCapabilities {
    let mut isa = IsaCapabilities::default();

    record(
        &mut isa,
        IsaFeature::Sse2,
        std::is_x86_feature_detected!("sse2"),
    );
    record(
        &mut isa,
        IsaFeature::Avx2,
        std::is_x86_feature_detected!("avx2"),
    );
    record(
        &mut isa,
        IsaFeature::Fma,
        std::is_x86_feature_detected!("fma"),
    );
    record(
        &mut isa,
        IsaFeature::Avx512F,
        std::is_x86_feature_detected!("avx512f"),
    );

    let max_vector_bits = if isa.supports(&IsaFeature::Avx512F)
    {
        Some(512)
    }
    else if isa.supports(&IsaFeature::Avx2)
    {
        Some(256)
    }
    else if isa.supports(&IsaFeature::Sse2)
    {
        Some(128)
    }
    else
    {
        None
    };

    if max_vector_bits.is_some()
    {
        isa.vector_model = VectorModel::FixedWidth;
        isa.min_vector_bits = Some(128);
        isa.max_vector_bits = max_vector_bits;
    }
    else
    {
        isa.vector_model = VectorModel::Scalar;
    }

    isa
}

#[cfg(target_arch = "aarch64")]
fn probe_aarch64() -> IsaCapabilities {
    let mut isa = IsaCapabilities::default();

    record(
        &mut isa,
        IsaFeature::Neon,
        std::arch::is_aarch64_feature_detected!("neon"),
    );
    record(
        &mut isa,
        IsaFeature::DotProd,
        std::arch::is_aarch64_feature_detected!("dotprod"),
    );
    record(
        &mut isa,
        IsaFeature::I8mm,
        std::arch::is_aarch64_feature_detected!("i8mm"),
    );
    record(
        &mut isa,
        IsaFeature::ArmBf16,
        std::arch::is_aarch64_feature_detected!("bf16"),
    );
    record(
        &mut isa,
        IsaFeature::Sve,
        std::arch::is_aarch64_feature_detected!("sve"),
    );
    record(
        &mut isa,
        IsaFeature::Sve2,
        std::arch::is_aarch64_feature_detected!("sve2"),
    );

    if isa.supports(&IsaFeature::Sve) || isa.supports(&IsaFeature::Sve2)
    {
        isa.vector_model = VectorModel::Scalable;
    }
    else if isa.supports(&IsaFeature::Neon)
    {
        isa.vector_model = VectorModel::FixedWidth;
        isa.min_vector_bits = Some(128);
        isa.max_vector_bits = Some(128);
    }
    else
    {
        isa.vector_model = VectorModel::Scalar;
    }

    isa
}

#[cfg(target_arch = "loongarch64")]
fn probe_loongarch64() -> IsaCapabilities {
    let mut isa = IsaCapabilities::default();

    record(
        &mut isa,
        IsaFeature::LoongArchLsx,
        std::arch::is_loongarch_feature_detected!("lsx"),
    );
    record(
        &mut isa,
        IsaFeature::LoongArchLasx,
        std::arch::is_loongarch_feature_detected!("lasx"),
    );

    if isa.supports(&IsaFeature::LoongArchLasx)
    {
        isa.vector_model = VectorModel::FixedWidth;
        isa.min_vector_bits = Some(128);
        isa.max_vector_bits = Some(256);
    }
    else if isa.supports(&IsaFeature::LoongArchLsx)
    {
        isa.vector_model = VectorModel::FixedWidth;
        isa.min_vector_bits = Some(128);
        isa.max_vector_bits = Some(128);
    }
    else
    {
        isa.vector_model = VectorModel::Scalar;
    }

    isa
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_identifies_the_host_cpu_without_inventing_backend_semantics() {
        let capabilities = probe_host_cpu();

        assert_eq!(capabilities.device, DeviceId::cpu());
        assert_eq!(capabilities.architecture, Architecture::current_host());
        assert!(capabilities.numeric.storage_dtypes.is_empty());
        assert_eq!(capabilities.matrix.accelerated, SupportLevel::Unknown);
        assert_eq!(
            capabilities.execution.async_execution,
            SupportLevel::Unknown
        );
        assert!(capabilities.reproducibility.modes.is_empty());
    }

    #[test]
    fn positive_and_negative_probe_results_are_disjoint() {
        let isa = probe_host_cpu().isa;

        for feature in isa.features.supported_values()
        {
            assert_eq!(isa.support_level(feature), SupportLevel::Supported);
        }
        for feature in isa.features.unsupported_values()
        {
            assert_eq!(isa.support_level(feature), SupportLevel::Unsupported);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_runtime_probe_resolves_the_supported_feature_set() {
        let isa = probe_host_cpu().isa;

        for feature in [
            IsaFeature::Sse2,
            IsaFeature::Avx2,
            IsaFeature::Fma,
            IsaFeature::Avx512F,
        ]
        {
            assert_ne!(isa.support_level(&feature), SupportLevel::Unknown);
        }
        assert_ne!(isa.vector_model, VectorModel::Unknown);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn aarch64_runtime_probe_resolves_stable_features() {
        let isa = probe_host_cpu().isa;

        for feature in [
            IsaFeature::Neon,
            IsaFeature::DotProd,
            IsaFeature::I8mm,
            IsaFeature::ArmBf16,
            IsaFeature::Sve,
            IsaFeature::Sve2,
        ]
        {
            assert_ne!(isa.support_level(&feature), SupportLevel::Unknown);
        }
        assert_ne!(isa.vector_model, VectorModel::Unknown);
        assert_eq!(isa.support_level(&IsaFeature::Sme), SupportLevel::Unknown);
        assert_eq!(isa.support_level(&IsaFeature::Sme2), SupportLevel::Unknown);
    }

    #[cfg(target_arch = "loongarch64")]
    #[test]
    fn loongarch_runtime_probe_resolves_lsx_and_lasx() {
        let isa = probe_host_cpu().isa;

        assert_ne!(
            isa.support_level(&IsaFeature::LoongArchLsx),
            SupportLevel::Unknown
        );
        assert_ne!(
            isa.support_level(&IsaFeature::LoongArchLasx),
            SupportLevel::Unknown
        );
        assert_ne!(isa.vector_model, VectorModel::Unknown);
    }

    #[cfg(target_arch = "riscv64")]
    #[test]
    fn riscv_vector_support_remains_unknown_without_a_stable_probe_contract() {
        let isa = probe_host_cpu().isa;

        assert_eq!(
            isa.support_level(&IsaFeature::RiscVVector),
            SupportLevel::Unknown
        );
        assert_eq!(isa.vector_model, VectorModel::Unknown);
    }
}
