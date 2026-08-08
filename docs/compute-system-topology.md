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

## Metrics and determinism

`TransferMetrics` is optional and records provenance as `Reported`, `Measured` or `Declared`.

Bandwidth and latency are metadata only. The deterministic capability planner must not silently change execution decisions based on fresh timing measurements. If benchmark-driven planning is introduced later, it requires an explicit policy and persisted provenance separate from the deterministic default planner.

## Validation

`SystemTopology::validate()` currently enforces structural invariants:

- node identifiers are unique;
- every link endpoint exists.

The graph deliberately does not reject cycles, multiple relations between the same nodes or missing performance data.

## Next implementation step

A host topology probe should populate this contract from reliable OS/backend interfaces. Linux NUMA/cache discovery should remain separate from accelerator-specific augmentation. Missing information must stay absent/unknown rather than being inferred from machine class or architecture name.
