//! Safe host topology probing for `SystemTopology`.
//!
//! The probe is a `std` concern. The topology data model itself remains
//! available in `no_std` builds.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::{
    CacheDescriptor, CacheKind, MemoryDomainDescriptor, MemorySpace, SupportLevel, SystemTopology,
    TopologyLink, TopologyNode, TopologyNodeId, TopologyNodeKind, TopologyRelation,
};

#[derive(Debug, Clone, PartialEq, Eq)]
struct CpuRecord {
    id: u32,
    package_id: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NumaRecord {
    id: u32,
    cpus: Vec<u32>,
    memory_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CacheKey {
    level: u8,
    kind_rank: u8,
    shared_cpus: Vec<u32>,
    fallback_cpu: Option<u32>,
    fallback_index: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CacheRecord {
    key: CacheKey,
    kind: CacheKind,
    size_bytes: Option<u64>,
    line_bytes: Option<u32>,
}

/// Probe the host's architecture-neutral compute topology.
///
/// On Linux the probe reads stable sysfs interfaces for online logical CPUs,
/// physical package identifiers, NUMA nodes, NUMA memory capacity and CPU cache
/// sharing. Missing or unreadable facts are omitted rather than inferred.
///
/// On non-Linux targets no equivalent portable contract is claimed yet, so an
/// empty, valid topology snapshot is returned.
pub fn probe_host_topology() -> SystemTopology {
    #[cfg(target_os = "linux")]
    {
        return probe_linux_topology(Path::new("/sys/devices/system"));
    }

    #[cfg(not(target_os = "linux"))]
    {
        SystemTopology::default()
    }
}

fn probe_linux_topology(system_root: &Path) -> SystemTopology {
    let cpu_root = system_root.join("cpu");
    let node_root = system_root.join("node");

    let cpus = collect_cpus(&cpu_root);
    let numa_nodes = collect_numa_nodes(&node_root);
    let caches = collect_caches(&cpu_root, &cpus);

    build_topology(&cpus, &numa_nodes, &caches)
}

fn collect_cpus(cpu_root: &Path) -> Vec<CpuRecord> {
    let online = read_trimmed(&cpu_root.join("online"))
        .and_then(|value| parse_cpu_list(&value))
        .map(|values| values.into_iter().collect::<BTreeSet<_>>());

    numeric_entries(cpu_root, "cpu")
        .into_iter()
        .filter(|(id, _)| online.as_ref().is_none_or(|set| set.contains(id)))
        .map(|(id, path)| CpuRecord {
            id,
            package_id: read_trimmed(&path.join("topology/physical_package_id"))
                .and_then(|value| value.parse::<u32>().ok()),
        })
        .collect()
}

fn collect_numa_nodes(node_root: &Path) -> Vec<NumaRecord> {
    numeric_entries(node_root, "node")
        .into_iter()
        .map(|(id, path)| NumaRecord {
            id,
            cpus: read_trimmed(&path.join("cpulist"))
                .and_then(|value| parse_cpu_list(&value))
                .unwrap_or_default(),
            memory_bytes: fs::read_to_string(path.join("meminfo"))
                .ok()
                .and_then(|value| parse_numa_mem_total(&value)),
        })
        .collect()
}

fn collect_caches(cpu_root: &Path, cpus: &[CpuRecord]) -> Vec<CacheRecord> {
    let mut caches = BTreeMap::<CacheKey, CacheRecord>::new();

    for cpu in cpus {
        let cache_root = cpu_root.join(format!("cpu{}/cache", cpu.id));
        for (index, path) in numeric_entries(&cache_root, "index") {
            let Some(level) = read_trimmed(&path.join("level"))
                .and_then(|value| value.parse::<u8>().ok())
            else {
                continue;
            };

            let kind = read_trimmed(&path.join("type"))
                .map(|value| parse_cache_kind(&value))
                .unwrap_or(CacheKind::Other);
            let shared = read_trimmed(&path.join("shared_cpu_list"))
                .and_then(|value| parse_cpu_list(&value));
            let (shared_cpus, fallback_cpu, fallback_index) = match shared {
                Some(cpus) if !cpus.is_empty() => (cpus, None, None),
                _ => (vec![cpu.id], Some(cpu.id), Some(index)),
            };

            let key = CacheKey {
                level,
                kind_rank: cache_kind_rank(kind),
                shared_cpus,
                fallback_cpu,
                fallback_index,
            };
            let record = CacheRecord {
                key: key.clone(),
                kind,
                size_bytes: read_trimmed(&path.join("size"))
                    .and_then(|value| parse_size_bytes(&value)),
                line_bytes: read_trimmed(&path.join("coherency_line_size"))
                    .and_then(|value| value.parse::<u32>().ok()),
            };
            caches.entry(key).or_insert(record);
        }
    }

    caches.into_values().collect()
}

fn build_topology(
    cpus: &[CpuRecord],
    numa_nodes: &[NumaRecord],
    caches: &[CacheRecord],
) -> SystemTopology {
    let mut topology = SystemTopology::default();
    let mut next_id = 0_u32;

    let machine = allocate_id(&mut next_id);
    let mut machine_node = TopologyNode::new(machine, TopologyNodeKind::Machine);
    machine_node.name = Some("machine".to_string());
    topology.nodes.push(machine_node);

    let mut package_ids = BTreeSet::new();
    for cpu in cpus {
        if let Some(package_id) = cpu.package_id {
            package_ids.insert(package_id);
        }
    }

    let mut package_nodes = BTreeMap::new();
    for package_id in package_ids {
        let node_id = allocate_id(&mut next_id);
        let mut node = TopologyNode::new(node_id, TopologyNodeKind::CpuPackage);
        node.name = Some(format!("cpu-package-{package_id}"));
        topology.nodes.push(node);
        topology.links.push(contains(machine, node_id));
        package_nodes.insert(package_id, node_id);
    }

    let mut numa_node_ids = BTreeMap::new();
    for numa in numa_nodes {
        let node_id = allocate_id(&mut next_id);
        let mut node = TopologyNode::new(node_id, TopologyNodeKind::NumaNode);
        node.name = Some(format!("numa-{}", numa.id));
        topology.nodes.push(node);
        topology.links.push(contains(machine, node_id));
        numa_node_ids.insert(numa.id, node_id);

        let memory_id = allocate_id(&mut next_id);
        let mut memory = TopologyNode::new(memory_id, TopologyNodeKind::MemoryDomain);
        memory.name = Some(format!("numa-{}-memory", numa.id));
        memory.memory = Some(MemoryDomainDescriptor {
            space: MemorySpace::Host,
            capacity_bytes: numa.memory_bytes,
            host_addressable: SupportLevel::Supported,
        });
        topology.nodes.push(memory);
        topology.links.push(TopologyLink {
            from: node_id,
            to: memory_id,
            relation: TopologyRelation::AffineTo,
            bidirectional: true,
            interconnect: None,
            metrics: None,
        });
    }

    let mut cpu_nodes = BTreeMap::new();
    for cpu in cpus {
        let node_id = allocate_id(&mut next_id);
        let mut node = TopologyNode::new(node_id, TopologyNodeKind::ProcessingUnit);
        node.name = Some(format!("cpu-{}", cpu.id));
        topology.nodes.push(node);

        let parent = cpu
            .package_id
            .and_then(|package_id| package_nodes.get(&package_id).copied())
            .unwrap_or(machine);
        topology.links.push(contains(parent, node_id));
        cpu_nodes.insert(cpu.id, node_id);
    }

    for numa in numa_nodes {
        let Some(numa_id) = numa_node_ids.get(&numa.id).copied() else {
            continue;
        };
        for cpu in &numa.cpus {
            if let Some(cpu_id) = cpu_nodes.get(cpu).copied() {
                topology.links.push(TopologyLink {
                    from: numa_id,
                    to: cpu_id,
                    relation: TopologyRelation::AffineTo,
                    bidirectional: true,
                    interconnect: None,
                    metrics: None,
                });
            }
        }
    }

    for cache in caches {
        let cache_id = allocate_id(&mut next_id);
        let mut node = TopologyNode::new(cache_id, TopologyNodeKind::Cache);
        node.name = Some(cache_name(cache));
        node.cache = Some(CacheDescriptor {
            level: cache.key.level,
            kind: cache.kind,
            size_bytes: cache.size_bytes,
            line_bytes: cache.line_bytes,
        });
        topology.nodes.push(node);

        let parent = common_package(cache, cpus, &package_nodes).unwrap_or(machine);
        topology.links.push(contains(parent, cache_id));

        for cpu in &cache.key.shared_cpus {
            if let Some(cpu_id) = cpu_nodes.get(cpu).copied() {
                topology.links.push(TopologyLink {
                    from: cache_id,
                    to: cpu_id,
                    relation: TopologyRelation::AffineTo,
                    bidirectional: true,
                    interconnect: None,
                    metrics: None,
                });
            }
        }
    }

    topology
}

fn common_package(
    cache: &CacheRecord,
    cpus: &[CpuRecord],
    package_nodes: &BTreeMap<u32, TopologyNodeId>,
) -> Option<TopologyNodeId> {
    let mut package = None;
    for cpu_id in &cache.key.shared_cpus {
        let current = cpus
            .iter()
            .find(|cpu| cpu.id == *cpu_id)
            .and_then(|cpu| cpu.package_id)?;
        match package {
            None => package = Some(current),
            Some(existing) if existing == current => {}
            Some(_) => return None,
        }
    }
    package.and_then(|id| package_nodes.get(&id).copied())
}

fn contains(from: TopologyNodeId, to: TopologyNodeId) -> TopologyLink {
    TopologyLink {
        from,
        to,
        relation: TopologyRelation::Contains,
        bidirectional: false,
        interconnect: None,
        metrics: None,
    }
}

fn allocate_id(next_id: &mut u32) -> TopologyNodeId {
    let id = TopologyNodeId::new(*next_id);
    *next_id = next_id.saturating_add(1);
    id
}

fn numeric_entries(root: &Path, prefix: &str) -> Vec<(u32, PathBuf)> {
    let Ok(entries) = fs::read_dir(root) else {
        return Vec::new();
    };

    let mut found = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let Some(suffix) = name.strip_prefix(prefix) else {
            continue;
        };
        let Ok(id) = suffix.parse::<u32>() else {
            continue;
        };
        found.push((id, entry.path()));
    }
    found.sort_by_key(|(id, _)| *id);
    found
}

fn read_trimmed(path: &Path) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_string())
}

fn parse_cpu_list(input: &str) -> Option<Vec<u32>> {
    let input = input.trim();
    if input.is_empty() {
        return Some(Vec::new());
    }

    let mut cpus = BTreeSet::new();
    for part in input.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return None;
        }

        if let Some((start, end)) = part.split_once('-') {
            if start.contains('-') || end.contains('-') {
                return None;
            }
            let start = start.parse::<u32>().ok()?;
            let end = end.parse::<u32>().ok()?;
            if start > end || end.saturating_sub(start) > 1_000_000 {
                return None;
            }
            for cpu in start..=end {
                cpus.insert(cpu);
            }
        } else {
            cpus.insert(part.parse::<u32>().ok()?);
        }
    }

    Some(cpus.into_iter().collect())
}

fn parse_cache_kind(input: &str) -> CacheKind {
    match input.trim() {
        "Data" => CacheKind::Data,
        "Instruction" => CacheKind::Instruction,
        "Unified" => CacheKind::Unified,
        _ => CacheKind::Other,
    }
}

fn cache_kind_rank(kind: CacheKind) -> u8 {
    match kind {
        CacheKind::Data => 0,
        CacheKind::Instruction => 1,
        CacheKind::Unified => 2,
        CacheKind::Other => 3,
    }
}

fn parse_size_bytes(input: &str) -> Option<u64> {
    let value = input.trim();
    if value.is_empty() {
        return None;
    }

    let split = value
        .find(|character: char| !character.is_ascii_digit())
        .unwrap_or(value.len());
    let number = value[..split].parse::<u64>().ok()?;
    let suffix = value[split..].trim();
    let multiplier = match suffix {
        "" | "B" => 1,
        "K" | "KB" | "kB" => 1024,
        "M" | "MB" => 1024_u64.pow(2),
        "G" | "GB" => 1024_u64.pow(3),
        _ => return None,
    };
    number.checked_mul(multiplier)
}

fn parse_numa_mem_total(input: &str) -> Option<u64> {
    for line in input.lines() {
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        let Some(index) = tokens.iter().position(|token| *token == "MemTotal:") else {
            continue;
        };
        let value = tokens.get(index + 1)?.parse::<u64>().ok()?;
        let unit = tokens.get(index + 2).copied().unwrap_or("B");
        let multiplier = match unit {
            "B" => 1,
            "kB" | "KB" | "K" => 1024,
            "MB" | "M" => 1024_u64.pow(2),
            "GB" | "G" => 1024_u64.pow(3),
            _ => return None,
        };
        return value.checked_mul(multiplier);
    }
    None
}

fn cache_name(cache: &CacheRecord) -> String {
    let kind = match cache.kind {
        CacheKind::Data => "data",
        CacheKind::Instruction => "instruction",
        CacheKind::Unified => "unified",
        CacheKind::Other => "other",
    };
    let cpus = cache
        .key
        .shared_cpus
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!("l{}-{kind}-cache-cpus-{cpus}", cache.key.level)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

    fn fixture_root() -> PathBuf {
        let id = FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "scirust-topology-probe-{}-{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn write(root: &Path, relative: &str, value: &str) {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, value).unwrap();
    }

    #[test]
    fn cpu_list_parser_is_sorted_deduplicated_and_rejects_invalid_ranges() {
        assert_eq!(parse_cpu_list("0"), Some(vec![0]));
        assert_eq!(
            parse_cpu_list("0-3,8,10-11"),
            Some(vec![0, 1, 2, 3, 8, 10, 11])
        );
        assert_eq!(parse_cpu_list("2,0-2"), Some(vec![0, 1, 2]));
        assert_eq!(parse_cpu_list("3-1"), None);
        assert_eq!(parse_cpu_list("0-1-2"), None);
        assert_eq!(parse_cpu_list("0,,2"), None);
    }

    #[test]
    fn cache_and_numa_sizes_use_binary_sysfs_units() {
        assert_eq!(parse_size_bytes("32K"), Some(32 * 1024));
        assert_eq!(parse_size_bytes("2M"), Some(2 * 1024 * 1024));
        assert_eq!(parse_size_bytes("invalid"), None);
        assert_eq!(
            parse_numa_mem_total("Node 0 MemTotal: 1024 kB\nNode 0 MemFree: 512 kB\n"),
            Some(1024 * 1024)
        );
    }

    #[test]
    fn synthetic_sysfs_builds_stable_numa_package_cpu_cache_topology() {
        let root = fixture_root();
        write(&root, "cpu/online", "0-1\n");
        write(&root, "cpu/cpu0/topology/physical_package_id", "0\n");
        write(&root, "cpu/cpu1/topology/physical_package_id", "0\n");
        write(&root, "node/node0/cpulist", "0-1\n");
        write(
            &root,
            "node/node0/meminfo",
            "Node 0 MemTotal: 65536 kB\n",
        );

        for cpu in 0..=1 {
            let prefix = format!("cpu/cpu{cpu}/cache/index0");
            write(&root, &format!("{prefix}/level"), "1\n");
            write(&root, &format!("{prefix}/type"), "Data\n");
            write(&root, &format!("{prefix}/size"), "32K\n");
            write(
                &root,
                &format!("{prefix}/coherency_line_size"),
                "64\n",
            );
            write(&root, &format!("{prefix}/shared_cpu_list"), "0-1\n");
        }

        let first = probe_linux_topology(&root);
        let second = probe_linux_topology(&root);
        assert_eq!(first, second);
        assert_eq!(first.validate(), Ok(()));
        assert_eq!(
            first
                .nodes
                .iter()
                .filter(|node| node.kind == TopologyNodeKind::Machine)
                .count(),
            1
        );
        assert_eq!(
            first
                .nodes
                .iter()
                .filter(|node| node.kind == TopologyNodeKind::CpuPackage)
                .count(),
            1
        );
        assert_eq!(
            first
                .nodes
                .iter()
                .filter(|node| node.kind == TopologyNodeKind::NumaNode)
                .count(),
            1
        );
        assert_eq!(
            first
                .nodes
                .iter()
                .filter(|node| node.kind == TopologyNodeKind::ProcessingUnit)
                .count(),
            2
        );
        assert_eq!(
            first
                .nodes
                .iter()
                .filter(|node| node.kind == TopologyNodeKind::Cache)
                .count(),
            1
        );
        assert_eq!(
            first
                .nodes
                .iter()
                .filter(|node| node.kind == TopologyNodeKind::MemoryDomain)
                .count(),
            1
        );

        let memory = first
            .nodes
            .iter()
            .find_map(|node| node.memory)
            .expect("NUMA memory domain");
        assert_eq!(memory.capacity_bytes, Some(64 * 1024 * 1024));
        assert_eq!(memory.host_addressable, SupportLevel::Supported);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn missing_sysfs_facts_are_omitted_instead_of_invented() {
        let root = fixture_root();
        write(&root, "cpu/online", "0\n");
        fs::create_dir_all(root.join("cpu/cpu0")).unwrap();

        let topology = probe_linux_topology(&root);
        assert_eq!(topology.validate(), Ok(()));
        assert_eq!(
            topology
                .nodes
                .iter()
                .filter(|node| node.kind == TopologyNodeKind::CpuPackage)
                .count(),
            0
        );
        assert_eq!(
            topology
                .nodes
                .iter()
                .filter(|node| node.kind == TopologyNodeKind::ProcessingUnit)
                .count(),
            1
        );
        assert_eq!(
            topology
                .nodes
                .iter()
                .filter(|node| node.kind == TopologyNodeKind::MemoryDomain)
                .count(),
            0
        );

        fs::remove_dir_all(root).unwrap();
    }
}
