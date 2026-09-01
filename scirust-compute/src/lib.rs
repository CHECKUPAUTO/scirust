//! Backend-neutral compute contracts for SciRust.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

mod backend;
mod backend_layers;
mod binding;
mod capabilities;
mod device;
mod dtype;
mod error;
mod hardware;
#[cfg(feature = "std")]
mod hwprobe;
mod ids;
mod implementation;
mod kernel;
mod launch;
mod memory;
mod profile_encoding;
mod requirements;
mod shape;
mod strides;
mod tensor;
mod topology;
mod transfer;
mod topology_augmentation;
#[cfg(feature = "std")]
mod topology_probe;
mod topology_provider;
mod workspace;

pub use backend::ComputeBackend;
pub use backend_layers::{
    BackendAllocator, BackendCompiler, BackendExecutor, BackendIntrospection, BackendRuntime,
};
pub use binding::{BufferAccess, BufferBinding};
pub use capabilities::DeviceCapabilities;
pub use device::{DeviceId, DeviceKind};
pub use dtype::DType;
pub use error::{ComputeError, ComputeResult};
pub use hardware::{
    Architecture, ArchitectureFamily, CapabilitySet, ExecutionCapabilities, HardwareCapabilities,
    IsaCapabilities, IsaFeature, MatrixCapabilities, MemoryCapabilities, NumericCapabilities,
    ReproducibilityCapabilities, ReproducibilityLevel, SupportLevel, VectorModel,
};
#[cfg(feature = "std")]
pub use hwprobe::probe_host_cpu;
pub use ids::{BufferId, EventId, KernelId, StreamId};
pub use implementation::{
    ExecutionLimits, ImplementationCandidate, ImplementationIssue, ImplementationMatchReport,
    ImplementationRequirements, ImplementationSelection, WorkgroupDimension, WorkgroupRequirement,
    match_implementation, select_implementation,
};
pub use kernel::{KernelFormat, KernelModule};
pub use launch::LaunchConfig;
pub use memory::{HostMemoryPolicy, HostPagePolicy, Layout, MemoryPolicyError, MemorySpace};
pub use profile_encoding::{
    ProfileEncodingError, canonical_hardware_profile_bytes, canonical_topology_profile_bytes,
};
pub use requirements::{
    CandidateSelection, ExecutionCandidate, KernelRequirements, MatchDisposition, MatchReport,
    PlannerPolicy, RequirementIssue, SupportRequirement, VectorRequirement, match_requirements,
    select_candidate,
};
pub use shape::Shape;
pub use strides::Strides;
pub use tensor::TensorSpec;
pub use transfer::{
    DTYPE_CONTRACT_VERSION, TRANSFER_CONTRACT_VERSION, TensorResidency, TransferMode,
    TransferRequest, TransferRequestError,
};
pub use topology::{
    CacheDescriptor, CacheKind, InterconnectClass, MemoryDomainDescriptor, MetricProvenance,
    SystemTopology, TopologyError, TopologyLink, TopologyNode, TopologyNodeId, TopologyNodeKind,
    TopologyRelation, TransferMetrics,
};
pub use topology_augmentation::{
    AcceleratorTopologyDescriptor, AcceleratorTopologyNodes, TopologyAugmentationError,
    augment_accelerator_topologies, augment_accelerator_topology,
};
#[cfg(feature = "std")]
pub use topology_probe::probe_host_topology;
pub use topology_provider::AcceleratorTopologyProvider;
pub use workspace::{KernelWorkspace, WorkspaceError, WorkspaceSpec};
