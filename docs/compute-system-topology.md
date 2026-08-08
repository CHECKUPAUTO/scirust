# SciRust System Topology Model

## Purpose

`HardwareCapabilities` answers **what one logical compute device can do**.
`SystemTopology` answers **where compute and memory resources are located relative to each other**.

The two contracts are intentionally separate. Architecture/ISA identity must not encode assumptions about sockets, NUMA, PCIe, unified memory or accelerator locality.

## Model

A topology snapshot is a graph of `TopologyNode` and `TopologyLink` values.

Nodes can represent:

- the machine;
- CPU packages;
- NUMA nodes;
- processing units;
- caches;
- memory domains;
- accelerators; and
- interconnect endpoints.

A node may carry a `DeviceId`, cache metadata or memory-domain metadata. This associates topology with the compute model without moving topology facts into `HardwareCapabilities`.

Links represent independent semantic relations:

- containment;
- affinity/locality;
- peer access;
- hardware coherence; and
- generic interconnect paths.

Cycles and multiple links between the same nodes are valid because these relations describe different facts.

## Architecture neutrality

The contract does not assume the classic `CPU -> PCIe -> discrete GPU` layout.

The same graph can describe, for example:

- a multisocket NUMA x86 server;
- an AArch64 SoC with shared/unified memory;
- a CPU with one or more discrete accelerators;
- coherent CXL-attached memory/accelerators;
- future processor/accelerator layouts not represented by today's enum variants.

Known interconnect classes are hints for classification, not dispatch identities. `Other` remains available for future interconnects.

## Host probing

With the `std` feature enabled, `probe_host_topology()` populates the model from host interfaces that SciRust can state conservatively.

On Linux the current probe reads sysfs for:

- online logical CPUs;
- `physical_package_id` package membership;
- NUMA node CPU lists;
- NUMA `MemTotal` capacity;
- cache level, role, size and coherency-line size;
- cache `shared_cpu_list` membership.

Directory enumeration, CPU lists, packages, NUMA nodes and cache identities are normalized into deterministic numeric order before topology node IDs are allocated. Shared caches are deduplicated from their reported shared CPU set rather than duplicated once per logical CPU.

The probe deliberately does **not** infer an interconnect class from CPU architecture, package identity or NUMA membership. A missing sysfs file, malformed value or unavailable concept is omitted instead of being converted into `Unsupported` or guessed from the machine class.

On non-Linux systems `probe_host_topology()` currently returns an empty valid snapshot rather than inventing a topology from incomplete portable APIs. The topology model itself remains available under `no_std`; only host probing requires `std`.

The Linux implementation is isolated behind target-specific compilation so Windows and macOS builds do not compile unused sysfs helpers under `-D warnings`.

## Metrics and determinism

`TransferMetrics` is optional and records provenance as `Reported`, `Measured` or `Declared`.

Bandwidth and latency are metadata only. The deterministic capability planner must not silently change execution decisions based on fresh timing measurements. If benchmark-driven planning is introduced later, it requires an explicit policy and persisted provenance separate from the deterministic default planner.

## Validation

`SystemTopology::validate()` currently enforces structural invariants:

- node identifiers are unique;
- every link endpoint exists.

The graph deliberately does not reject cycles, multiple relations between the same nodes or missing performance data.

The Linux host-probe tests use synthetic sysfs fixtures rather than depending on the CI machine's real topology. They verify deterministic snapshots, shared-cache deduplication, NUMA memory parsing and conservative behavior when sysfs facts are absent.

## Next implementation step

Accelerator backends should augment the host snapshot with device nodes and only those locality, memory-domain, peer-access or interconnect facts that their runtime APIs can establish explicitly. CPU, WGPU and CUDA backend-specific `HardwareCapabilities` should remain separate from this topology augmentation.
