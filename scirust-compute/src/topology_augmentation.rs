extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use crate::{
    DeviceId, DeviceKind, MemoryDomainDescriptor, SystemTopology, TopologyError, TopologyLink,
    TopologyNode, TopologyNodeId, TopologyNodeKind, TopologyRelation,
};

/// Backend-proven facts that can be added to a [`SystemTopology`] snapshot.
///
/// This descriptor intentionally contains no PCIe/on-chip/interconnect field.
/// An accelerator backend may know that a logical device and a logical memory
/// domain belong together without knowing the physical transfer fabric. Such a
/// fabric must be added later only when a backend or OS interface proves it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceleratorTopologyDescriptor {
    pub device: DeviceId,
    pub name: Option<String>,
    pub memory: Option<MemoryDomainDescriptor>,
}

impl AcceleratorTopologyDescriptor {
    pub const fn new(device: DeviceId) -> Self {
        Self {
            device,
            name: None,
            memory: None,
        }
    }
}

/// Stable node identifiers allocated by one accelerator augmentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcceleratorTopologyNodes {
    pub accelerator: TopologyNodeId,
    pub memory: Option<TopologyNodeId>,
}

/// Failure while adding backend-proven accelerator facts to a topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TopologyAugmentationError {
    InvalidTopology(TopologyError),
    DuplicateDevice(DeviceId),
    NodeIdExhausted,
}

/// Add one logical accelerator, and optionally its logical memory domain, to an
/// existing topology snapshot.
///
/// This is the single-device convenience wrapper around
/// [`augment_accelerator_topologies`]. No physical interconnect, transfer metric
/// or coherence relation is inferred.
pub fn augment_accelerator_topology(
    topology: &mut SystemTopology,
    descriptor: AcceleratorTopologyDescriptor,
) -> Result<AcceleratorTopologyNodes, TopologyAugmentationError> {
    let added = augment_accelerator_topologies(topology, core::slice::from_ref(&descriptor))?;
    debug_assert_eq!(added.len(), 1);
    Ok(added[0])
}

/// Add a set of logical accelerators in deterministic device order.
///
/// The input order does not affect node identifiers. Descriptors are sorted by
/// a stable `(DeviceKind, ordinal)` key before IDs are allocated, so backend
/// discovery order cannot perturb a topology fingerprint. The update is
/// transactional: invalid topology, duplicate devices or node-ID exhaustion
/// leave the original topology unchanged.
///
/// When a descriptor contains a logical memory domain, a bidirectional
/// [`TopologyRelation::AffineTo`] link is added between the accelerator and that
/// memory domain. The link deliberately carries no interconnect class or
/// transfer metrics.
pub fn augment_accelerator_topologies(
    topology: &mut SystemTopology,
    descriptors: &[AcceleratorTopologyDescriptor],
) -> Result<Vec<AcceleratorTopologyNodes>, TopologyAugmentationError> {
    topology
        .validate()
        .map_err(TopologyAugmentationError::InvalidTopology)?;

    let mut ordered = descriptors.to_vec();
    ordered.sort_by_key(|descriptor| device_sort_key(descriptor.device));

    if let Some(duplicate) = ordered
        .windows(2)
        .find(|pair| pair[0].device == pair[1].device)
        .map(|pair| pair[0].device)
    {
        return Err(TopologyAugmentationError::DuplicateDevice(duplicate));
    }

    for descriptor in &ordered
    {
        if topology
            .nodes
            .iter()
            .any(|node| node.device == Some(descriptor.device))
        {
            return Err(TopologyAugmentationError::DuplicateDevice(
                descriptor.device,
            ));
        }
    }

    let mut candidate = topology.clone();
    let mut added = Vec::with_capacity(ordered.len());

    for descriptor in ordered
    {
        added.push(append_accelerator_unchecked(&mut candidate, descriptor)?);
    }

    candidate
        .validate()
        .map_err(TopologyAugmentationError::InvalidTopology)?;
    *topology = candidate;

    Ok(added)
}

fn append_accelerator_unchecked(
    topology: &mut SystemTopology,
    descriptor: AcceleratorTopologyDescriptor,
) -> Result<AcceleratorTopologyNodes, TopologyAugmentationError> {
    let mut used_ids = topology
        .nodes
        .iter()
        .map(|node| node.id.get())
        .collect::<Vec<_>>();
    used_ids.sort_unstable();
    used_ids.dedup();

    let accelerator =
        first_free_node_id(&used_ids).ok_or(TopologyAugmentationError::NodeIdExhausted)?;
    used_ids.push(accelerator.get());
    used_ids.sort_unstable();

    let memory_id = if descriptor.memory.is_some()
    {
        Some(
            first_free_node_id(&used_ids)
                .ok_or(TopologyAugmentationError::NodeIdExhausted)?,
        )
    }
    else
    {
        None
    };

    let mut accelerator_node = TopologyNode::new(accelerator, TopologyNodeKind::Accelerator);
    accelerator_node.name = descriptor.name;
    accelerator_node.device = Some(descriptor.device);
    topology.nodes.push(accelerator_node);

    if let (Some(memory), Some(memory_id)) = (descriptor.memory, memory_id)
    {
        let mut memory_node = TopologyNode::new(memory_id, TopologyNodeKind::MemoryDomain);
        memory_node.memory = Some(memory);
        topology.nodes.push(memory_node);
        topology.links.push(TopologyLink {
            from: accelerator,
            to: memory_id,
            relation: TopologyRelation::AffineTo,
            bidirectional: true,
            interconnect: None,
            metrics: None,
        });
    }

    Ok(AcceleratorTopologyNodes {
        accelerator,
        memory: memory_id,
    })
}

fn device_sort_key(device: DeviceId) -> (u8, u32) {
    let kind = match device.kind()
    {
        DeviceKind::Reference => 0,
        DeviceKind::Cpu => 1,
        DeviceKind::Wgpu => 2,
        DeviceKind::Cuda => 3,
    };
    (kind, device.ordinal())
}

fn first_free_node_id(used_ids: &[u32]) -> Option<TopologyNodeId> {
    let mut candidate = 0_u32;

    for &used in used_ids
    {
        if used < candidate
        {
            continue;
        }
        if used > candidate
        {
            return Some(TopologyNodeId::new(candidate));
        }

        candidate = candidate.checked_add(1)?;
    }

    Some(TopologyNodeId::new(candidate))
}

#[cfg(test)]
mod tests {
    use alloc::{format, vec};

    use super::*;
    use crate::{MemorySpace, SupportLevel};

    fn cuda_descriptor(ordinal: u32) -> AcceleratorTopologyDescriptor {
        let mut descriptor =
            AcceleratorTopologyDescriptor::new(DeviceId::new(DeviceKind::Cuda, ordinal));
        descriptor.name = Some(format!("cuda-{ordinal}"));
        descriptor.memory = Some(MemoryDomainDescriptor {
            space: MemorySpace::Device,
            capacity_bytes: Some(24 * 1024 * 1024 * 1024),
            host_addressable: SupportLevel::Unknown,
        });
        descriptor
    }

    #[test]
    fn augmentation_uses_smallest_free_ids_and_no_fabric_inference() {
        let machine = TopologyNode::new(TopologyNodeId::new(0), TopologyNodeKind::Machine);
        let numa = TopologyNode::new(TopologyNodeId::new(2), TopologyNodeKind::NumaNode);
        let mut topology = SystemTopology {
            nodes: vec![machine, numa],
            links: Vec::new(),
        };

        let added = augment_accelerator_topology(&mut topology, cuda_descriptor(0)).unwrap();

        assert_eq!(added.accelerator, TopologyNodeId::new(1));
        assert_eq!(added.memory, Some(TopologyNodeId::new(3)));
        assert_eq!(topology.validate(), Ok(()));

        let link = topology
            .links
            .iter()
            .find(|link| link.from == added.accelerator)
            .expect("accelerator-memory affinity");
        assert_eq!(link.to, added.memory.unwrap());
        assert_eq!(link.relation, TopologyRelation::AffineTo);
        assert!(link.bidirectional);
        assert_eq!(link.interconnect, None);
        assert_eq!(link.metrics, None);
    }

    #[test]
    fn batch_augmentation_is_independent_of_discovery_order() {
        let descriptors = [
            cuda_descriptor(1),
            AcceleratorTopologyDescriptor::new(DeviceId::new(DeviceKind::Wgpu, 0)),
            cuda_descriptor(0),
        ];

        let mut forward = SystemTopology::default();
        let forward_added = augment_accelerator_topologies(&mut forward, &descriptors).unwrap();

        let mut reversed_descriptors = descriptors.to_vec();
        reversed_descriptors.reverse();
        let mut reversed = SystemTopology::default();
        let reversed_added =
            augment_accelerator_topologies(&mut reversed, &reversed_descriptors).unwrap();

        assert_eq!(forward_added, reversed_added);
        assert_eq!(forward, reversed);
        assert_eq!(forward.validate(), Ok(()));
    }

    #[test]
    fn accelerator_without_memory_adds_no_synthetic_memory_or_link() {
        let mut topology = SystemTopology::default();
        let descriptor =
            AcceleratorTopologyDescriptor::new(DeviceId::new(DeviceKind::Wgpu, 0));

        let added = augment_accelerator_topology(&mut topology, descriptor).unwrap();

        assert_eq!(added.accelerator, TopologyNodeId::new(0));
        assert_eq!(added.memory, None);
        assert_eq!(topology.nodes.len(), 1);
        assert!(topology.links.is_empty());
    }

    #[test]
    fn duplicate_logical_device_is_rejected_without_mutation() {
        let device = DeviceId::new(DeviceKind::Cuda, 0);
        let mut existing = TopologyNode::new(TopologyNodeId::new(4), TopologyNodeKind::Accelerator);
        existing.device = Some(device);
        let mut topology = SystemTopology {
            nodes: vec![existing],
            links: Vec::new(),
        };
        let before = topology.clone();

        assert_eq!(
            augment_accelerator_topology(
                &mut topology,
                AcceleratorTopologyDescriptor::new(device)
            ),
            Err(TopologyAugmentationError::DuplicateDevice(device))
        );
        assert_eq!(topology, before);
    }

    #[test]
    fn duplicate_batch_device_is_rejected_transactionally() {
        let device = DeviceId::new(DeviceKind::Cuda, 0);
        let descriptors = [
            AcceleratorTopologyDescriptor::new(device),
            AcceleratorTopologyDescriptor::new(device),
        ];
        let mut topology = SystemTopology {
            nodes: vec![TopologyNode::new(
                TopologyNodeId::new(7),
                TopologyNodeKind::Machine,
            )],
            links: Vec::new(),
        };
        let before = topology.clone();

        assert_eq!(
            augment_accelerator_topologies(&mut topology, &descriptors),
            Err(TopologyAugmentationError::DuplicateDevice(device))
        );
        assert_eq!(topology, before);
    }

    #[test]
    fn invalid_input_topology_is_rejected_without_mutation() {
        let duplicate = TopologyNodeId::new(7);
        let mut topology = SystemTopology {
            nodes: vec![
                TopologyNode::new(duplicate, TopologyNodeKind::Machine),
                TopologyNode::new(duplicate, TopologyNodeKind::NumaNode),
            ],
            links: Vec::new(),
        };
        let before = topology.clone();

        assert_eq!(
            augment_accelerator_topology(
                &mut topology,
                AcceleratorTopologyDescriptor::new(DeviceId::new(DeviceKind::Wgpu, 0))
            ),
            Err(TopologyAugmentationError::InvalidTopology(
                TopologyError::DuplicateNode(duplicate)
            ))
        );
        assert_eq!(topology, before);
    }

    #[test]
    fn first_free_id_fills_gaps_deterministically() {
        assert_eq!(first_free_node_id(&[]), Some(TopologyNodeId::new(0)));
        assert_eq!(
            first_free_node_id(&[0, 1, 3, 4]),
            Some(TopologyNodeId::new(2))
        );
        assert_eq!(first_free_node_id(&[1, 2]), Some(TopologyNodeId::new(0)));
        assert_eq!(first_free_node_id(&[u32::MAX]), Some(TopologyNodeId::new(0)));
    }
}
