//! Safe host topology probing for `SystemTopology`.
//!
//! The probe is a `std` concern. The topology data model itself remains
//! available in `no_std` builds.

#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

use crate::SystemTopology;

#[cfg(target_os = "linux")]
mod linux;

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
        linux::probe()
    }

    #[cfg(not(target_os = "linux"))]
    {
        SystemTopology::default()
    }
}
