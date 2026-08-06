# Elastic Latent KV — Phase 11 Lifecycle

Phase 11 adds fixed-capacity token lifecycle control. A deterministic ring tracks
logical position, basis version, access tick and HOT/WARM/COLD temperature.

Admission evicts the oldest resident token when capacity is full. Rebalancing is
based on logical recency and writes re-encoding actions into caller-owned scratch,
so the controller performs no allocation after construction.

Each temperature maps to a compression target containing coefficient format,
residual format, residual-slot cap and rank divisor. This separates lifecycle
policy from the scalar Phase 8 codec and from the Phase 9 planning objective.
