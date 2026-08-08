extern crate alloc;

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use crate::{DType, DeviceId, HardwareCapabilities};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceCapabilities {
    pub device: DeviceId,
    pub name: String,
    pub supported_dtypes: Vec<DType>,
    pub max_buffer_bytes: Option<usize>,
    pub max_workgroup_size: [u32; 3],
    pub supports_async_execution: bool,
}

impl DeviceCapabilities {
    pub fn reference_cpu() -> Self {
        Self {
            device: DeviceId::reference(),
            name: "reference-cpu".to_string(),
            supported_dtypes: vec![DType::F32],
            max_buffer_bytes: None,
            max_workgroup_size: [1, 1, 1],
            supports_async_execution: false,
        }
    }

    pub fn supports_dtype(&self, dtype: DType) -> bool {
        self.supported_dtypes.contains(&dtype)
    }

    /// Conservative architecture-neutral hardware view derived from the facts
    /// represented by this legacy capability structure.
    ///
    /// Optional ISA, matrix, memory and reproducibility properties remain
    /// unknown until a backend or hardware probe advertises them explicitly.
    pub fn hardware_baseline(&self) -> HardwareCapabilities {
        HardwareCapabilities::from_device_capabilities(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SupportLevel, VectorModel};

    #[test]
    fn reference_capabilities_are_honest() {
        let capabilities = DeviceCapabilities::reference_cpu();

        assert_eq!(capabilities.device, DeviceId::reference());
        assert!(capabilities.supports_dtype(DType::F32));
        assert!(!capabilities.supports_dtype(DType::F64));
        assert!(!capabilities.supports_async_execution);
    }

    #[test]
    fn hardware_baseline_does_not_invent_optional_features() {
        let hardware = DeviceCapabilities::reference_cpu().hardware_baseline();

        assert!(hardware.isa.features.is_empty());
        assert_eq!(hardware.isa.vector_model, VectorModel::Unknown);
        assert_eq!(
            hardware.execution.async_execution,
            SupportLevel::Unsupported
        );
        assert_eq!(hardware.matrix.accelerated, SupportLevel::Unknown);
    }
}
