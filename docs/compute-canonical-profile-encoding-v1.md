# Canonical compute profile encoding v1

Execution attestation needs stable fingerprints of the compute facts that justified backend selection. `Debug` output, JSON field order, vector insertion order, and runtime enumeration order are not suitable fingerprint inputs.

`scirust-compute` therefore exposes canonical **bytes**, not a hash algorithm:

- `canonical_hardware_profile_bytes(&HardwareCapabilities)`;
- `canonical_topology_profile_bytes(&SystemTopology)`.

The execution-attestation layer can hash these bytes with its protocol-defined SHA-256 implementation without introducing a dependency from `scirust-compute` back to the agent protocol.

## Hardware capabilities

The v1 encoding is domain-separated with `scirust.hardware-capabilities.v1\0` and includes the complete capability snapshot:

- logical `DeviceId`;
- architecture family and optional semantic name;
- ISA feature tri-state partitions and vector model/bounds;
- storage, arithmetic and accumulation dtype partitions;
- matrix acceleration and dtype partitions;
- memory-space partitions and coherence/addressing/transfer support levels;
- execution semantics;
- reproducibility modes.

Every enum uses an explicit numeric tag. Capability-set supported and unsupported partitions are encoded separately and sorted by canonical value bytes before emission. Consequently insertion order does not affect the profile identity, while `Unknown` remains distinct from explicitly `Unsupported`.

## System topology

The v1 topology encoding is domain-separated with `scirust.system-topology.v1\0`.

The graph is validated before encoding. Invalid graphs fail closed with `ProfileEncodingError::InvalidTopology`.

Canonicalization normalizes:

- node-vector order by `TopologyNodeId`;
- link-vector order by each link's canonical bytes;
- endpoint orientation for links explicitly marked bidirectional.

Node IDs themselves remain part of the identity. All node descriptors, logical device identities, cache/memory facts, relations, interconnect classes and optional transfer metrics are encoded when present.

A change from omitted/unknown information to a reported fact therefore changes the topology identity. Performance metrics remain metadata rather than planner truth, but when they are present in the attested snapshot their values and provenance are fingerprinted.

## Stability

The encoding uses fixed-order fields, explicit enum tags, presence bytes for optional values, little-endian fixed-width integers, and length-prefixed UTF-8 text/byte strings.

Changing an enum tag or field order changes the v1 identity and must not be done accidentally. Tests verify insertion-order independence, topology-order independence, bidirectional-link normalization, `Unknown` versus `Unsupported`, metric sensitivity, and invalid-topology rejection.

## Portability

The implementation is `alloc`-only and remains compatible with `scirust-compute --no-default-features`. The permanent compute topology gate includes this module and verifies no-std compatibility at Rust 1.89.
