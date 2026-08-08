extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use crate::{DeviceId, MemorySpace, SupportLevel};

/// Stable identifier of a node in one [`SystemTopology`] snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TopologyNodeId(u32);

impl TopologyNodeId {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Kind of entity represented by a topology node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TopologyNodeKind {
    Machine,
    CpuPackage,
    NumaNode,
    ProcessingUnit,
    Cache,
    MemoryDomain,
    Accelerator,
    InterconnectEndpoint,
    Other,
}

/// Cache role when a topology node represents a cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CacheKind {
    Data,
    Instruction,
    Unified,
    Other,
}

/// Optional cache description attached to a topology node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheDescriptor {
    pub level: u8,
    pub kind: CacheKind,
    pub size_bytes: Option<u64>,
    pub line_bytes: Option<u32>,
}

/// Optional memory-domain description attached to a topology node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryDomainDescriptor {
    pub space: MemorySpace,
    pub capacity_bytes: Option<u64>,
    /// Whether the domain can be directly addressed by the host CPU.
    pub host_addressable: SupportLevel,
}

/// One node in a topology snapshot.
///
/// A node may refer to a logical compute device, but the topology remains a
/// separate contract: architecture/ISA capabilities live in
/// `HardwareCapabilities`, while this graph describes locality and containment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyNode {
    pub id: TopologyNodeId,
    pub kind: TopologyNodeKind,
    pub name: Option<String>,
    pub device: Option<DeviceId>,
    pub cache: Option<CacheDescriptor>,
    pub memory: Option<MemoryDomainDescriptor>,
}

impl TopologyNode {
    pub fn new(id: TopologyNodeId, kind: TopologyNodeKind) -> Self {
        Self {
            id,
            kind,
            name: None,
            device: None,
            cache: None,
            memory: None,
        }
    }
}

/// Semantic relation between two topology nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TopologyRelation {
    /// `from` physically or logically contains `to`.
    Contains,
    /// A compute entity is local/affine to another entity or memory domain.
    AffineTo,
    /// The endpoints can access each other without host staging.
    PeerAccess,
    /// The endpoints share a hardware-coherent domain.
    CoherentWith,
    /// Generic data-transfer/interconnect path.
    Interconnect,
    Other,
}

/// Broad interconnect class without assuming one vendor or machine layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum InterconnectClass {
    OnChip,
    SharedMemory,
    HostMemoryBus,
    Pcie,
    Cxl,
    HighSpeedPeer,
    NetworkFabric,
    Other,
}

/// Provenance of optional performance data attached to a topology link.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MetricProvenance {
    /// Reported by firmware, an OS interface or a backend API.
    Reported,
    /// Measured by an explicit benchmark/probe.
    Measured,
    /// Declared by a user or deployment manifest.
    Declared,
}

/// Optional transfer characteristics for a topology link.
///
/// These values are metadata, never capability truth. Planners may ignore them;
/// deterministic selection must not silently depend on fresh timing measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransferMetrics {
    pub bandwidth_bytes_per_second: Option<u64>,
    pub latency_nanoseconds: Option<u64>,
    pub provenance: MetricProvenance,
}

/// One directed topology relation. Set `bidirectional` when the same relation is
/// known to hold in both directions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyLink {
    pub from: TopologyNodeId,
    pub to: TopologyNodeId,
    pub relation: TopologyRelation,
    pub bidirectional: bool,
    pub interconnect: Option<InterconnectClass>,
    pub metrics: Option<TransferMetrics>,
}

/// Validation failure for a topology snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TopologyError {
    DuplicateNode(TopologyNodeId),
    MissingEndpoint(TopologyNodeId),
}

/// Architecture-neutral snapshot of machine locality and memory topology.
///
/// The graph intentionally allows cycles and multiple links between the same
/// nodes because coherence, affinity and transfer paths are independent facts.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SystemTopology {
    pub nodes: Vec<TopologyNode>,
    pub links: Vec<TopologyLink>,
}

impl SystemTopology {
    pub fn validate(&self) -> Result<(), TopologyError> {
        for (index, node) in self.nodes.iter().enumerate()
        {
            if self.nodes[index + 1..]
                .iter()
                .any(|candidate| candidate.id == node.id)
            {
                return Err(TopologyError::DuplicateNode(node.id));
            }
        }

        for link in &self.links
        {
            if !self.contains_node(link.from)
            {
                return Err(TopologyError::MissingEndpoint(link.from));
            }
            if !self.contains_node(link.to)
            {
                return Err(TopologyError::MissingEndpoint(link.to));
            }
        }

        Ok(())
    }

    pub fn contains_node(&self, id: TopologyNodeId) -> bool {
        self.nodes.iter().any(|node| node.id == id)
    }

    pub fn node(&self, id: TopologyNodeId) -> Option<&TopologyNode> {
        self.nodes.iter().find(|node| node.id == id)
    }

    pub fn links_from(&self, id: TopologyNodeId) -> impl Iterator<Item = &TopologyLink> {
        self.links.iter().filter(move |link| link.from == id)
    }

    /// Return links usable from `id`, including reverse traversal for links
    /// explicitly marked bidirectional.
    pub fn adjacent_links(&self, id: TopologyNodeId) -> impl Iterator<Item = &TopologyLink> {
        self.links
            .iter()
            .filter(move |link| link.from == id || (link.bidirectional && link.to == id))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::{DeviceKind, MemorySpace};

    #[test]
    fn topology_accepts_cpu_numa_accelerator_locality() {
        let machine = TopologyNodeId::new(0);
        let numa = TopologyNodeId::new(1);
        let memory = TopologyNodeId::new(2);
        let gpu = TopologyNodeId::new(3);

        let mut topology = SystemTopology {
            nodes: vec![
                TopologyNode::new(machine, TopologyNodeKind::Machine),
                TopologyNode::new(numa, TopologyNodeKind::NumaNode),
                TopologyNode {
                    memory: Some(MemoryDomainDescriptor {
                        space: MemorySpace::Host,
                        capacity_bytes: Some(64 * 1024 * 1024 * 1024),
                        host_addressable: SupportLevel::Supported,
                    }),
                    ..TopologyNode::new(memory, TopologyNodeKind::MemoryDomain)
                },
                TopologyNode {
                    device: Some(DeviceId::new(DeviceKind::Cuda, 0)),
                    ..TopologyNode::new(gpu, TopologyNodeKind::Accelerator)
                },
            ],
            links: vec![
                TopologyLink {
                    from: machine,
                    to: numa,
                    relation: TopologyRelation::Contains,
                    bidirectional: false,
                    interconnect: None,
                    metrics: None,
                },
                TopologyLink {
                    from: numa,
                    to: memory,
                    relation: TopologyRelation::AffineTo,
                    bidirectional: true,
                    interconnect: Some(InterconnectClass::HostMemoryBus),
                    metrics: None,
                },
                TopologyLink {
                    from: numa,
                    to: gpu,
                    relation: TopologyRelation::AffineTo,
                    bidirectional: true,
                    interconnect: Some(InterconnectClass::Pcie),
                    metrics: None,
                },
            ],
        };

        assert_eq!(topology.validate(), Ok(()));
        assert_eq!(topology.adjacent_links(gpu).count(), 1);
        assert_eq!(topology.links_from(numa).count(), 2);

        topology.links[2].metrics = Some(TransferMetrics {
            bandwidth_bytes_per_second: Some(24_000_000_000),
            latency_nanoseconds: None,
            provenance: MetricProvenance::Measured,
        });
        assert_eq!(topology.validate(), Ok(()));
    }

    #[test]
    fn duplicate_node_ids_are_rejected() {
        let id = TopologyNodeId::new(7);
        let topology = SystemTopology {
            nodes: vec![
                TopologyNode::new(id, TopologyNodeKind::CpuPackage),
                TopologyNode::new(id, TopologyNodeKind::NumaNode),
            ],
            links: Vec::new(),
        };

        assert_eq!(topology.validate(), Err(TopologyError::DuplicateNode(id)));
    }

    #[test]
    fn links_to_missing_nodes_are_rejected() {
        let existing = TopologyNodeId::new(1);
        let missing = TopologyNodeId::new(99);
        let topology = SystemTopology {
            nodes: vec![TopologyNode::new(existing, TopologyNodeKind::Machine)],
            links: vec![TopologyLink {
                from: existing,
                to: missing,
                relation: TopologyRelation::Contains,
                bidirectional: false,
                interconnect: None,
                metrics: None,
            }],
        };

        assert_eq!(
            topology.validate(),
            Err(TopologyError::MissingEndpoint(missing))
        );
    }

    #[test]
    fn topology_does_not_require_transfer_metrics() {
        let a = TopologyNodeId::new(1);
        let b = TopologyNodeId::new(2);
        let topology = SystemTopology {
            nodes: vec![
                TopologyNode::new(a, TopologyNodeKind::MemoryDomain),
                TopologyNode::new(b, TopologyNodeKind::Accelerator),
            ],
            links: vec![TopologyLink {
                from: a,
                to: b,
                relation: TopologyRelation::Interconnect,
                bidirectional: true,
                interconnect: None,
                metrics: None,
            }],
        };

        assert_eq!(topology.validate(), Ok(()));
    }
}
