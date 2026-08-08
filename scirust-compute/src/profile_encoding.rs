extern crate alloc;

use alloc::{vec, vec::Vec};

use crate::{
    ArchitectureFamily, CacheDescriptor, CacheKind, CapabilitySet, DType, DeviceId, DeviceKind,
    HardwareCapabilities, InterconnectClass, IsaFeature, MemoryDomainDescriptor, MemorySpace,
    MetricProvenance, ReproducibilityLevel, SupportLevel, SystemTopology, TopologyError,
    TopologyLink, TopologyNode, TopologyNodeKind, TopologyRelation, TransferMetrics, VectorModel,
};

const HARDWARE_PROFILE_DOMAIN: &[u8] = b"scirust.hardware-capabilities.v1\0";
const TOPOLOGY_PROFILE_DOMAIN: &[u8] = b"scirust.system-topology.v1\0";

/// Failure while creating a deterministic compute-profile encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProfileEncodingError {
    InvalidTopology(TopologyError),
    LengthOverflow,
}

/// Canonical v1 representation of [`HardwareCapabilities`].
///
/// Capability-set insertion order is normalized before encoding. The returned
/// bytes are suitable as input to a stable content digest, but this function
/// intentionally does not select a hashing algorithm or depend on an agent/wire
/// protocol crate.
pub fn canonical_hardware_profile_bytes(
    hardware: &HardwareCapabilities,
) -> Result<Vec<u8>, ProfileEncodingError> {
    let mut out = Vec::with_capacity(512);
    out.extend_from_slice(HARDWARE_PROFILE_DOMAIN);
    put_device(&mut out, hardware.device);
    out.push(architecture_family_tag(hardware.architecture.family));
    put_optional_text(&mut out, hardware.architecture.name.as_deref())?;

    put_capability_set(&mut out, &hardware.isa.features, encode_isa_feature)?;
    out.push(vector_model_tag(hardware.isa.vector_model));
    put_optional_u32(&mut out, hardware.isa.min_vector_bits);
    put_optional_u32(&mut out, hardware.isa.max_vector_bits);

    put_capability_set(&mut out, &hardware.numeric.storage_dtypes, encode_dtype)?;
    put_capability_set(&mut out, &hardware.numeric.arithmetic_dtypes, encode_dtype)?;
    put_capability_set(
        &mut out,
        &hardware.numeric.accumulation_dtypes,
        encode_dtype,
    )?;

    out.push(support_tag(hardware.matrix.accelerated));
    put_capability_set(&mut out, &hardware.matrix.input_dtypes, encode_dtype)?;
    put_capability_set(&mut out, &hardware.matrix.accumulation_dtypes, encode_dtype)?;

    put_capability_set(&mut out, &hardware.memory.spaces, encode_memory_space)?;
    out.push(support_tag(hardware.memory.coherent_host_device));
    out.push(support_tag(hardware.memory.unified_addressing));
    out.push(support_tag(hardware.memory.async_transfers));

    out.push(support_tag(hardware.execution.async_execution));
    out.push(support_tag(hardware.execution.ordered_streams));
    out.push(support_tag(hardware.execution.subgroup_operations));
    out.push(support_tag(hardware.execution.atomic_i64));

    put_capability_set(
        &mut out,
        &hardware.reproducibility.modes,
        encode_reproducibility,
    )?;

    Ok(out)
}

/// Canonical v1 representation of [`SystemTopology`].
///
/// Node-vector order, link-vector order, and the endpoint orientation of a link
/// explicitly marked bidirectional do not affect the encoding. Node identifiers
/// themselves remain part of the topology identity.
pub fn canonical_topology_profile_bytes(
    topology: &SystemTopology,
) -> Result<Vec<u8>, ProfileEncodingError> {
    topology
        .validate()
        .map_err(ProfileEncodingError::InvalidTopology)?;

    let mut out = Vec::with_capacity(512);
    out.extend_from_slice(TOPOLOGY_PROFILE_DOMAIN);

    let mut nodes = topology.nodes.iter().collect::<Vec<_>>();
    nodes.sort_by_key(|node| node.id);
    put_len(&mut out, nodes.len())?;
    for node in nodes
    {
        encode_node(&mut out, node)?;
    }

    let mut links = topology
        .links
        .iter()
        .map(encode_link)
        .collect::<Result<Vec<_>, _>>()?;
    links.sort();
    put_len(&mut out, links.len())?;
    for link in links
    {
        put_bytes(&mut out, &link)?;
    }

    Ok(out)
}

fn encode_node(out: &mut Vec<u8>, node: &TopologyNode) -> Result<(), ProfileEncodingError> {
    put_u32(out, node.id.get());
    out.push(topology_node_kind_tag(node.kind));
    put_optional_text(out, node.name.as_deref())?;
    put_optional_device(out, node.device);
    put_optional_cache(out, node.cache);
    put_optional_memory_domain(out, node.memory);
    Ok(())
}

fn encode_link(link: &TopologyLink) -> Result<Vec<u8>, ProfileEncodingError> {
    let (from, to) = if link.bidirectional && link.to < link.from
    {
        (link.to, link.from)
    }
    else
    {
        (link.from, link.to)
    };

    let mut out = Vec::with_capacity(64);
    put_u32(&mut out, from.get());
    put_u32(&mut out, to.get());
    out.push(topology_relation_tag(link.relation));
    out.push(u8::from(link.bidirectional));
    put_optional_interconnect(&mut out, link.interconnect);
    put_optional_metrics(&mut out, link.metrics);
    Ok(out)
}

fn put_capability_set<T: PartialEq, F>(
    out: &mut Vec<u8>,
    set: &CapabilitySet<T>,
    mut encode: F,
) -> Result<(), ProfileEncodingError>
where
    F: FnMut(&T) -> Result<Vec<u8>, ProfileEncodingError>,
{
    let mut supported = set
        .supported_values()
        .iter()
        .map(&mut encode)
        .collect::<Result<Vec<_>, _>>()?;
    supported.sort();
    put_len(out, supported.len())?;
    for value in supported
    {
        put_bytes(out, &value)?;
    }

    let mut unsupported = set
        .unsupported_values()
        .iter()
        .map(&mut encode)
        .collect::<Result<Vec<_>, _>>()?;
    unsupported.sort();
    put_len(out, unsupported.len())?;
    for value in unsupported
    {
        put_bytes(out, &value)?;
    }
    Ok(())
}

fn encode_dtype(dtype: &DType) -> Result<Vec<u8>, ProfileEncodingError> {
    Ok(vec![dtype_tag(*dtype)])
}

fn encode_memory_space(space: &MemorySpace) -> Result<Vec<u8>, ProfileEncodingError> {
    Ok(vec![memory_space_tag(*space)])
}

fn encode_reproducibility(level: &ReproducibilityLevel) -> Result<Vec<u8>, ProfileEncodingError> {
    Ok(vec![reproducibility_tag(*level)])
}

fn encode_isa_feature(feature: &IsaFeature) -> Result<Vec<u8>, ProfileEncodingError> {
    let mut out = Vec::new();
    match feature
    {
        IsaFeature::Sse2 => out.push(0),
        IsaFeature::Avx2 => out.push(1),
        IsaFeature::Fma => out.push(2),
        IsaFeature::Avx512F => out.push(3),
        IsaFeature::Avx512Vnni => out.push(4),
        IsaFeature::Avx512Bf16 => out.push(5),
        IsaFeature::Avx512Fp16 => out.push(6),
        IsaFeature::AmxTile => out.push(7),
        IsaFeature::AmxInt8 => out.push(8),
        IsaFeature::AmxBf16 => out.push(9),
        IsaFeature::Neon => out.push(10),
        IsaFeature::DotProd => out.push(11),
        IsaFeature::I8mm => out.push(12),
        IsaFeature::ArmBf16 => out.push(13),
        IsaFeature::Sve => out.push(14),
        IsaFeature::Sve2 => out.push(15),
        IsaFeature::Sme => out.push(16),
        IsaFeature::Sme2 => out.push(17),
        IsaFeature::RiscVVector => out.push(18),
        IsaFeature::LoongArchLsx => out.push(19),
        IsaFeature::LoongArchLasx => out.push(20),
        IsaFeature::Other(name) =>
        {
            out.push(21);
            put_text(&mut out, name)?;
        },
    }
    Ok(out)
}

fn put_optional_device(out: &mut Vec<u8>, device: Option<DeviceId>) {
    match device
    {
        Some(device) =>
        {
            out.push(1);
            put_device(out, device);
        },
        None => out.push(0),
    }
}

fn put_device(out: &mut Vec<u8>, device: DeviceId) {
    out.push(device_kind_tag(device.kind()));
    put_u32(out, device.ordinal());
}

fn put_optional_cache(out: &mut Vec<u8>, cache: Option<CacheDescriptor>) {
    match cache
    {
        Some(cache) =>
        {
            out.push(1);
            out.push(cache.level);
            out.push(cache_kind_tag(cache.kind));
            put_optional_u64(out, cache.size_bytes);
            put_optional_u32(out, cache.line_bytes);
        },
        None => out.push(0),
    }
}

fn put_optional_memory_domain(out: &mut Vec<u8>, memory: Option<MemoryDomainDescriptor>) {
    match memory
    {
        Some(memory) =>
        {
            out.push(1);
            out.push(memory_space_tag(memory.space));
            put_optional_u64(out, memory.capacity_bytes);
            out.push(support_tag(memory.host_addressable));
        },
        None => out.push(0),
    }
}

fn put_optional_interconnect(out: &mut Vec<u8>, value: Option<InterconnectClass>) {
    match value
    {
        Some(value) =>
        {
            out.push(1);
            out.push(interconnect_tag(value));
        },
        None => out.push(0),
    }
}

fn put_optional_metrics(out: &mut Vec<u8>, metrics: Option<TransferMetrics>) {
    match metrics
    {
        Some(metrics) =>
        {
            out.push(1);
            put_optional_u64(out, metrics.bandwidth_bytes_per_second);
            put_optional_u64(out, metrics.latency_nanoseconds);
            out.push(metric_provenance_tag(metrics.provenance));
        },
        None => out.push(0),
    }
}

fn put_optional_text(out: &mut Vec<u8>, value: Option<&str>) -> Result<(), ProfileEncodingError> {
    match value
    {
        Some(value) =>
        {
            out.push(1);
            put_text(out, value)?;
        },
        None => out.push(0),
    }
    Ok(())
}

fn put_text(out: &mut Vec<u8>, value: &str) -> Result<(), ProfileEncodingError> {
    put_bytes(out, value.as_bytes())
}

fn put_bytes(out: &mut Vec<u8>, value: &[u8]) -> Result<(), ProfileEncodingError> {
    put_len(out, value.len())?;
    out.extend_from_slice(value);
    Ok(())
}

fn put_len(out: &mut Vec<u8>, value: usize) -> Result<(), ProfileEncodingError> {
    let value = u32::try_from(value).map_err(|_| ProfileEncodingError::LengthOverflow)?;
    put_u32(out, value);
    Ok(())
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_optional_u32(out: &mut Vec<u8>, value: Option<u32>) {
    match value
    {
        Some(value) =>
        {
            out.push(1);
            put_u32(out, value);
        },
        None => out.push(0),
    }
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_optional_u64(out: &mut Vec<u8>, value: Option<u64>) {
    match value
    {
        Some(value) =>
        {
            out.push(1);
            put_u64(out, value);
        },
        None => out.push(0),
    }
}

fn support_tag(value: SupportLevel) -> u8 {
    match value
    {
        SupportLevel::Unknown => 0,
        SupportLevel::Unsupported => 1,
        SupportLevel::Supported => 2,
    }
}

fn device_kind_tag(value: DeviceKind) -> u8 {
    match value
    {
        DeviceKind::Reference => 0,
        DeviceKind::Cpu => 1,
        DeviceKind::Wgpu => 2,
        DeviceKind::Cuda => 3,
    }
}

fn architecture_family_tag(value: ArchitectureFamily) -> u8 {
    match value
    {
        ArchitectureFamily::Unknown => 0,
        ArchitectureFamily::X86_64 => 1,
        ArchitectureFamily::Aarch64 => 2,
        ArchitectureFamily::RiscV64 => 3,
        ArchitectureFamily::LoongArch64 => 4,
        ArchitectureFamily::Wasm32 => 5,
        ArchitectureFamily::NvidiaGpu => 6,
        ArchitectureFamily::AmdGpu => 7,
        ArchitectureFamily::IntelGpu => 8,
        ArchitectureFamily::AppleGpu => 9,
        ArchitectureFamily::Other => 10,
    }
}

fn vector_model_tag(value: VectorModel) -> u8 {
    match value
    {
        VectorModel::Unknown => 0,
        VectorModel::Scalar => 1,
        VectorModel::FixedWidth => 2,
        VectorModel::Scalable => 3,
    }
}

fn dtype_tag(value: DType) -> u8 {
    match value
    {
        DType::Bool => 0,
        DType::U8 => 1,
        DType::I8 => 2,
        DType::U16 => 3,
        DType::I16 => 4,
        DType::F16 => 5,
        DType::Bf16 => 6,
        DType::U32 => 7,
        DType::I32 => 8,
        DType::F32 => 9,
        DType::U64 => 10,
        DType::I64 => 11,
        DType::F64 => 12,
    }
}

fn memory_space_tag(value: MemorySpace) -> u8 {
    match value
    {
        MemorySpace::Host => 0,
        MemorySpace::HostPinned => 1,
        MemorySpace::Device => 2,
        MemorySpace::Unified => 3,
    }
}

fn reproducibility_tag(value: ReproducibilityLevel) -> u8 {
    match value
    {
        ReproducibilityLevel::BitExact => 0,
        ReproducibilityLevel::Deterministic => 1,
        ReproducibilityLevel::NumericallyEquivalent => 2,
        ReproducibilityLevel::FastApproximate => 3,
    }
}

fn topology_node_kind_tag(value: TopologyNodeKind) -> u8 {
    match value
    {
        TopologyNodeKind::Machine => 0,
        TopologyNodeKind::CpuPackage => 1,
        TopologyNodeKind::NumaNode => 2,
        TopologyNodeKind::ProcessingUnit => 3,
        TopologyNodeKind::Cache => 4,
        TopologyNodeKind::MemoryDomain => 5,
        TopologyNodeKind::Accelerator => 6,
        TopologyNodeKind::InterconnectEndpoint => 7,
        TopologyNodeKind::Other => 8,
    }
}

fn cache_kind_tag(value: CacheKind) -> u8 {
    match value
    {
        CacheKind::Data => 0,
        CacheKind::Instruction => 1,
        CacheKind::Unified => 2,
        CacheKind::Other => 3,
    }
}

fn topology_relation_tag(value: TopologyRelation) -> u8 {
    match value
    {
        TopologyRelation::Contains => 0,
        TopologyRelation::AffineTo => 1,
        TopologyRelation::PeerAccess => 2,
        TopologyRelation::CoherentWith => 3,
        TopologyRelation::Interconnect => 4,
        TopologyRelation::Other => 5,
    }
}

fn interconnect_tag(value: InterconnectClass) -> u8 {
    match value
    {
        InterconnectClass::OnChip => 0,
        InterconnectClass::SharedMemory => 1,
        InterconnectClass::HostMemoryBus => 2,
        InterconnectClass::Pcie => 3,
        InterconnectClass::Cxl => 4,
        InterconnectClass::HighSpeedPeer => 5,
        InterconnectClass::NetworkFabric => 6,
        InterconnectClass::Other => 7,
    }
}

fn metric_provenance_tag(value: MetricProvenance) -> u8 {
    match value
    {
        MetricProvenance::Reported => 0,
        MetricProvenance::Measured => 1,
        MetricProvenance::Declared => 2,
    }
}

#[cfg(test)]
mod tests {
    use alloc::{string::ToString, vec};

    use super::*;
    use crate::{Architecture, DeviceCapabilities};

    #[test]
    fn hardware_encoding_ignores_capability_insertion_order() {
        let mut left =
            HardwareCapabilities::from_device_capabilities(&DeviceCapabilities::reference_cpu());
        left.architecture = Architecture::named(ArchitectureFamily::X86_64, "test-x86");
        left.isa
            .features
            .set_support(IsaFeature::Avx2, SupportLevel::Supported);
        left.isa
            .features
            .set_support(IsaFeature::Fma, SupportLevel::Supported);

        let mut right =
            HardwareCapabilities::from_device_capabilities(&DeviceCapabilities::reference_cpu());
        right.architecture = Architecture::named(ArchitectureFamily::X86_64, "test-x86");
        right
            .isa
            .features
            .set_support(IsaFeature::Fma, SupportLevel::Supported);
        right
            .isa
            .features
            .set_support(IsaFeature::Avx2, SupportLevel::Supported);

        assert_eq!(
            canonical_hardware_profile_bytes(&left).unwrap(),
            canonical_hardware_profile_bytes(&right).unwrap()
        );
    }

    #[test]
    fn hardware_encoding_preserves_unknown_vs_unsupported() {
        let unknown =
            HardwareCapabilities::from_device_capabilities(&DeviceCapabilities::reference_cpu());
        let mut unsupported = unknown.clone();
        unsupported
            .numeric
            .arithmetic_dtypes
            .set_support(DType::F64, SupportLevel::Unsupported);

        assert_ne!(
            canonical_hardware_profile_bytes(&unknown).unwrap(),
            canonical_hardware_profile_bytes(&unsupported).unwrap()
        );
    }

    #[test]
    fn topology_encoding_ignores_vector_order_and_bidirectional_orientation() {
        let machine = TopologyNode::new(crate::TopologyNodeId::new(0), TopologyNodeKind::Machine);
        let mut accelerator =
            TopologyNode::new(crate::TopologyNodeId::new(1), TopologyNodeKind::Accelerator);
        accelerator.name = Some("gpu".to_string());
        accelerator.device = Some(DeviceId::new(DeviceKind::Cuda, 0));

        let forward = TopologyLink {
            from: machine.id,
            to: accelerator.id,
            relation: TopologyRelation::AffineTo,
            bidirectional: true,
            interconnect: None,
            metrics: None,
        };
        let reverse = TopologyLink {
            from: accelerator.id,
            to: machine.id,
            ..forward.clone()
        };

        let left = SystemTopology {
            nodes: vec![machine.clone(), accelerator.clone()],
            links: vec![forward],
        };
        let right = SystemTopology {
            nodes: vec![accelerator, machine],
            links: vec![reverse],
        };

        assert_eq!(
            canonical_topology_profile_bytes(&left).unwrap(),
            canonical_topology_profile_bytes(&right).unwrap()
        );
    }

    #[test]
    fn topology_metrics_remain_part_of_identity() {
        let a = crate::TopologyNodeId::new(0);
        let b = crate::TopologyNodeId::new(1);
        let nodes = vec![
            TopologyNode::new(a, TopologyNodeKind::Machine),
            TopologyNode::new(b, TopologyNodeKind::Accelerator),
        ];
        let base = TopologyLink {
            from: a,
            to: b,
            relation: TopologyRelation::Interconnect,
            bidirectional: false,
            interconnect: Some(InterconnectClass::Pcie),
            metrics: None,
        };
        let without_metrics = SystemTopology {
            nodes: nodes.clone(),
            links: vec![base.clone()],
        };
        let with_metrics = SystemTopology {
            nodes,
            links: vec![TopologyLink {
                metrics: Some(TransferMetrics {
                    bandwidth_bytes_per_second: Some(24_000_000_000),
                    latency_nanoseconds: None,
                    provenance: MetricProvenance::Reported,
                }),
                ..base
            }],
        };

        assert_ne!(
            canonical_topology_profile_bytes(&without_metrics).unwrap(),
            canonical_topology_profile_bytes(&with_metrics).unwrap()
        );
    }

    #[test]
    fn invalid_topology_fails_closed() {
        let topology = SystemTopology {
            nodes: vec![TopologyNode::new(
                crate::TopologyNodeId::new(0),
                TopologyNodeKind::Machine,
            )],
            links: vec![TopologyLink {
                from: crate::TopologyNodeId::new(0),
                to: crate::TopologyNodeId::new(9),
                relation: TopologyRelation::Contains,
                bidirectional: false,
                interconnect: None,
                metrics: None,
            }],
        };

        assert_eq!(
            canonical_topology_profile_bytes(&topology),
            Err(ProfileEncodingError::InvalidTopology(
                TopologyError::MissingEndpoint(crate::TopologyNodeId::new(9))
            ))
        );
    }
}
