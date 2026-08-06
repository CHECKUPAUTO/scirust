/// Family of compute backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DeviceKind {
    Reference,
    Cpu,
    Wgpu,
    Cuda,
}

/// Stable logical identifier of a compute device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId {
    kind: DeviceKind,
    ordinal: u32,
}

impl DeviceId {
    pub const fn new(kind: DeviceKind, ordinal: u32) -> Self {
        Self { kind, ordinal }
    }

    pub const fn reference() -> Self {
        Self::new(DeviceKind::Reference, 0)
    }

    pub const fn cpu() -> Self {
        Self::new(DeviceKind::Cpu, 0)
    }

    pub const fn kind(self) -> DeviceKind {
        self.kind
    }

    pub const fn ordinal(self) -> u32 {
        self.ordinal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_devices_use_ordinal_zero() {
        assert_eq!(DeviceId::reference().ordinal(), 0);
        assert_eq!(DeviceId::cpu().ordinal(), 0);
    }

    #[test]
    fn device_identity_includes_kind_and_ordinal() {
        assert_ne!(
            DeviceId::new(DeviceKind::Cuda, 0),
            DeviceId::new(DeviceKind::Cuda, 1)
        );
        assert_ne!(
            DeviceId::new(DeviceKind::Cpu, 0),
            DeviceId::new(DeviceKind::Cuda, 0)
        );
    }
}
