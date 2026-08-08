//! Backend-neutral compute contracts for SciRust.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

mod backend;
mod binding;
mod capabilities;
mod device;
mod dtype;
mod error;
mod hardware;
#[cfg(feature = "std")]
mod hwprobe;
mod ids;
mod kernel;
mod launch;
mod memory;
mod requirements;
mod shape;
mod strides;
mod tensor;
mod topology;
#[cfg(feature = "std")]
mod topology_probe;

pub use backend::ComputeBackend;
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
pub use kernel::{KernelFormat, KernelModule};
pub use launch::LaunchConfig;
pub use memory::{Layout, MemorySpace};
pub use requirements::{
    CandidateSelection, ExecutionCandidate, KernelRequirements, MatchDisposition, MatchReport,
    PlannerPolicy, RequirementIssue, SupportRequirement, VectorRequirement, match_requirements,
    select_candidate,
};
pub use shape::Shape;
pub use strides::Strides;
pub use tensor::TensorSpec;
pub use topology::{
    CacheDescriptor, CacheKind, InterconnectClass, MemoryDomainDescriptor, MetricProvenance,
    SystemTopology, TopologyError, TopologyLink, TopologyNode, TopologyNodeId, TopologyNodeKind,
    TopologyRelation, TransferMetrics,
};
#[cfg(feature = "std")]
pub use topology_probe::probe_host_topology;
