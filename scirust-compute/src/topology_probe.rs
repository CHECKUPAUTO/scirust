//! Safe host topology probing for `SystemTopology`.
//!
//! The probe is a `std` concern. The topology data model itself remains
//! available in `no_std` builds.

#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

#[cfg(target_os = "linux")]
use std::collections::BTreeSet;
#[cfg(target_os = "linux")]
use std::fs;

use crate::SystemTopology;
#[cfg(target_os = "linux")]
use crate::TopologyNodeKind;

#[cfg(target_os = "linux")]
mod linux;

/// Probe the host's architecture-neutral compute topology.
///
/// On Linux the probe reads stable sysfs interfaces for online logical CPUs,
/// physical package identifiers, NUMA nodes, NUMA memory capacity and CPU cache
/// sharing. Missing or unreadable facts are omitted rather than inferred.
///
/// The public probe fails closed when Linux cannot establish the online CPU set:
/// no processing-unit snapshot is returned from directory enumeration alone.
///
/// On non-Linux targets no equivalent portable contract is claimed yet, so an
/// empty, valid topology snapshot is returned.
pub fn probe_host_topology() -> SystemTopology {
    #[cfg(target_os = "linux")]
    {
        let Some(online) = linux_online_cpu_ids()
        else
        {
            return SystemTopology::default();
        };

        let topology = linux::probe();
        if topology_has_cpu_outside_online_set(&topology, &online)
        {
            return SystemTopology::default();
        }
        topology
    }

    #[cfg(not(target_os = "linux"))]
    {
        SystemTopology::default()
    }
}

#[cfg(target_os = "linux")]
fn linux_online_cpu_ids() -> Option<BTreeSet<u32>> {
    let value = fs::read_to_string("/sys/devices/system/cpu/online").ok()?;
    parse_linux_cpu_list(&value)
}

#[cfg(target_os = "linux")]
fn topology_has_cpu_outside_online_set(
    topology: &SystemTopology,
    online: &BTreeSet<u32>,
) -> bool {
    topology
        .nodes
        .iter()
        .filter(|node| node.kind == TopologyNodeKind::ProcessingUnit)
        .any(|node| {
            let Some(id) = node
                .name
                .as_deref()
                .and_then(|name| name.strip_prefix("cpu-"))
                .and_then(|id| id.parse::<u32>().ok())
            else
            {
                return true;
            };
            !online.contains(&id)
        })
}

#[cfg(target_os = "linux")]
fn parse_linux_cpu_list(input: &str) -> Option<BTreeSet<u32>> {
    let input = input.trim();
    if input.is_empty()
    {
        return Some(BTreeSet::new());
    }

    let mut cpus = BTreeSet::new();
    for part in input.split(',')
    {
        let part = part.trim();
        if part.is_empty()
        {
            return None;
        }

        if let Some((start, end)) = part.split_once('-')
        {
            if start.contains('-') || end.contains('-')
            {
                return None;
            }
            let start = start.parse::<u32>().ok()?;
            let end = end.parse::<u32>().ok()?;
            if start > end || end.saturating_sub(start) > 1_000_000
            {
                return None;
            }
            for cpu in start..=end
            {
                cpus.insert(cpu);
            }
        }
        else
        {
            cpus.insert(part.parse::<u32>().ok()?);
        }
    }

    Some(cpus)
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use crate::{TopologyNode, TopologyNodeId};

    #[test]
    fn online_cpu_parser_is_sorted_deduplicated_and_fail_closed() {
        assert_eq!(
            parse_linux_cpu_list("0-2,5,2"),
            Some(BTreeSet::from([0, 1, 2, 5]))
        );
        assert_eq!(parse_linux_cpu_list("3-1"), None);
        assert_eq!(parse_linux_cpu_list("0,,2"), None);
    }

    #[test]
    fn processing_units_outside_online_set_are_rejected() {
        let mut topology = SystemTopology::default();
        let mut cpu0 = TopologyNode::new(TopologyNodeId::new(0), TopologyNodeKind::ProcessingUnit);
        cpu0.name = Some("cpu-0".into());
        topology.nodes.push(cpu0);

        assert!(!topology_has_cpu_outside_online_set(
            &topology,
            &BTreeSet::from([0])
        ));
        assert!(topology_has_cpu_outside_online_set(
            &topology,
            &BTreeSet::from([1])
        ));
    }

    #[test]
    fn unnamed_processing_unit_is_rejected_conservatively() {
        let mut topology = SystemTopology::default();
        topology.nodes.push(TopologyNode::new(
            TopologyNodeId::new(0),
            TopologyNodeKind::ProcessingUnit,
        ));

        assert!(topology_has_cpu_outside_online_set(
            &topology,
            &BTreeSet::from([0])
        ));
    }
}
