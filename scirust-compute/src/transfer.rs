extern crate alloc;

use alloc::format;
use alloc::string::String;

use crate::{DType, DeviceId, DeviceKind, MemorySpace, StreamId};

/// Version of the backend-neutral tensor-transfer contract.
pub const TRANSFER_CONTRACT_VERSION: u16 = 1;

/// Version of the dtype names used by canonical transfer records.
pub const DTYPE_CONTRACT_VERSION: u16 = 1;

/// Where a tensor's storage is resident.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TensorResidency {
    /// Ordinary pageable host memory.
    Host,
    /// Host memory prepared for efficient device transfers.
    HostPinned,
    /// Storage owned by one identified compute device.
    Device(DeviceId),
    /// Storage in a unified domain associated with one identified device.
    Unified(DeviceId),
}

impl TensorResidency {
    /// Memory-space category represented by this residency.
    pub const fn memory_space(self) -> MemorySpace {
        match self {
            Self::Host => MemorySpace::Host,
            Self::HostPinned => MemorySpace::HostPinned,
            Self::Device(_) => MemorySpace::Device,
            Self::Unified(_) => MemorySpace::Unified,
        }
    }

    /// Device identity, when this residency is associated with a device.
    pub const fn device(self) -> Option<DeviceId> {
        match self {
            Self::Host | Self::HostPinned => None,
            Self::Device(device) | Self::Unified(device) => Some(device),
        }
    }

    fn canonical_record(self) -> String {
        match self {
            Self::Host => String::from("host"),
            Self::HostPinned => String::from("host-pinned"),
            Self::Device(device) => format!(
                "device:{}:{}",
                device_kind_name(device.kind()),
                device.ordinal()
            ),
            Self::Unified(device) => format!(
                "unified:{}:{}",
                device_kind_name(device.kind()),
                device.ordinal()
            ),
        }
    }
}

/// Execution ordering requested for a transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TransferMode {
    /// The transfer is complete when the backend call returns.
    Synchronous,
    /// The transfer is ordered on the declared stream and may be in flight.
    Asynchronous {
        /// Stream that owns the transfer ordering.
        stream: StreamId,
    },
}

impl TransferMode {
    /// Stream associated with an asynchronous transfer.
    pub const fn stream(self) -> Option<StreamId> {
        match self {
            Self::Synchronous => None,
            Self::Asynchronous { stream } => Some(stream),
        }
    }

    /// Whether the request has asynchronous completion semantics.
    pub const fn is_asynchronous(self) -> bool {
        matches!(self, Self::Asynchronous { .. })
    }
}

/// A validated, backend-neutral tensor transfer request.
///
/// This is a contract only: constructing a request does not allocate memory,
/// stage through host memory, or claim that a backend supports the requested
/// path. A backend must either execute the declared source/destination pair or
/// return an explicit unsupported result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransferRequest {
    source: TensorResidency,
    destination: TensorResidency,
    dtype: DType,
    byte_len: usize,
    mode: TransferMode,
}

impl TransferRequest {
    /// Construct a transfer request with explicit residency, dtype and ordering.
    pub const fn new(
        source: TensorResidency,
        destination: TensorResidency,
        dtype: DType,
        byte_len: usize,
        mode: TransferMode,
    ) -> Result<Self, TransferRequestError> {
        if source == destination {
            return Err(TransferRequestError::SameResidency { residency: source });
        }

        Ok(Self {
            source,
            destination,
            dtype,
            byte_len,
            mode,
        })
    }

    /// Source residency requested by the caller.
    pub const fn source(self) -> TensorResidency {
        self.source
    }

    /// Destination residency requested by the caller.
    pub const fn destination(self) -> TensorResidency {
        self.destination
    }

    /// Scalar type carried by the transfer.
    pub const fn dtype(self) -> DType {
        self.dtype
    }

    /// Number of bytes represented by the transfer.
    pub const fn byte_len(self) -> usize {
        self.byte_len
    }

    /// Ordering/completion semantics requested by the caller.
    pub const fn mode(self) -> TransferMode {
        self.mode
    }

    /// Stable record for provenance and cross-backend diagnostics.
    #[must_use]
    pub fn canonical_record(self) -> String {
        let stream = match self.mode {
            TransferMode::Synchronous => String::from("none"),
            TransferMode::Asynchronous { stream } => stream.get().to_string(),
        };
        format!(
            "scirust-transfer-v{TRANSFER_CONTRACT_VERSION};dtype_version={DTYPE_CONTRACT_VERSION};dtype={};bytes={};source={};destination={};mode={};stream={stream}",
            dtype_name(self.dtype),
            self.byte_len,
            self.source.canonical_record(),
            self.destination.canonical_record(),
            mode_name(self.mode),
        )
    }
}

/// Construction failure for a transfer request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferRequestError {
    /// A transfer between identical residency values would be an implicit no-op.
    SameResidency {
        /// Residency that was repeated.
        residency: TensorResidency,
    },
}

impl core::fmt::Display for TransferRequestError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::SameResidency { residency } => write!(
                formatter,
                "transfer source and destination are identical: {residency:?}"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for TransferRequestError {}

const fn mode_name(mode: TransferMode) -> &'static str {
    match mode {
        TransferMode::Synchronous => "sync",
        TransferMode::Asynchronous { .. } => "async",
    }
}

const fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "bool",
        DType::U8 => "u8",
        DType::I8 => "i8",
        DType::U16 => "u16",
        DType::I16 => "i16",
        DType::F16 => "f16",
        DType::Bf16 => "bf16",
        DType::U32 => "u32",
        DType::I32 => "i32",
        DType::F32 => "f32",
        DType::U64 => "u64",
        DType::I64 => "i64",
        DType::F64 => "f64",
    }
}

const fn device_kind_name(kind: DeviceKind) -> &'static str {
    match kind {
        DeviceKind::Reference => "reference",
        DeviceKind::Cpu => "cpu",
        DeviceKind::Wgpu => "wgpu",
        DeviceKind::Cuda => "cuda",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residency_preserves_explicit_device_identity() {
        let device = DeviceId::new(DeviceKind::Wgpu, 2);
        let residency = TensorResidency::Device(device);

        assert_eq!(residency.memory_space(), MemorySpace::Device);
        assert_eq!(residency.device(), Some(device));
    }

    #[test]
    fn asynchronous_request_records_declared_stream_without_fallback() {
        let request = TransferRequest::new(
            TensorResidency::HostPinned,
            TensorResidency::Device(DeviceId::new(DeviceKind::Cuda, 1)),
            DType::F32,
            4096,
            TransferMode::Asynchronous {
                stream: StreamId::new(7),
            },
        )
        .expect("distinct residencies");

        assert!(request.mode().is_asynchronous());
        assert_eq!(request.mode().stream(), Some(StreamId::new(7)));
        assert_eq!(
            request.canonical_record(),
            "scirust-transfer-v1;dtype_version=1;dtype=f32;bytes=4096;source=host-pinned;destination=device:cuda:1;mode=async;stream=7"
        );
    }

    #[test]
    fn synchronous_request_has_no_hidden_stream() {
        let request = TransferRequest::new(
            TensorResidency::Device(DeviceId::cpu()),
            TensorResidency::Unified(DeviceId::cpu()),
            DType::F64,
            8,
            TransferMode::Synchronous,
        )
        .expect("distinct residencies");

        assert!(!request.mode().is_asynchronous());
        assert_eq!(request.mode().stream(), None);
        assert_eq!(
            request.canonical_record(),
            "scirust-transfer-v1;dtype_version=1;dtype=f64;bytes=8;source=device:cpu:0;destination=unified:cpu:0;mode=sync;stream=none"
        );
    }

    #[test]
    fn identical_residency_is_rejected_explicitly() {
        let residency = TensorResidency::Device(DeviceId::cpu());

        assert_eq!(
            TransferRequest::new(
                residency,
                residency,
                DType::F32,
                4,
                TransferMode::Synchronous
            ),
            Err(TransferRequestError::SameResidency { residency })
        );
    }
}
