use crate::AcceleratorTopologyDescriptor;

/// Backend-neutral source of logical accelerator-topology facts.
///
/// Implementations must report only facts established by the acquired backend
/// instance. In particular, this trait does not authorize inferring physical
/// interconnects, coherence, NUMA affinity, peer access, or transfer metrics
/// from an API name, vendor string, or allocation space.
pub trait AcceleratorTopologyProvider {
    /// Describe the logical accelerator and optional memory domain exposed by
    /// this backend instance.
    fn accelerator_topology_descriptor(&self) -> AcceleratorTopologyDescriptor;
}
