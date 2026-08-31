# SciRust Agent Bootstrap Contract

This repository has a persistent agent-oriented ecosystem roadmap that is intentionally kept off the default branch.

## Mandatory first step

Before any autonomous coding, shared API change, tensor/representation change, cross-repository integration, architectural decision, PR creation, or merge decision, read:

`origin/agent/ecosystem-roadmap:.agent/SCIRUST_ECOSYSTEM_ROADMAP.yaml`

Recommended command:

```bash
git fetch origin agent/ecosystem-roadmap && \
git show origin/agent/ecosystem-roadmap:.agent/SCIRUST_ECOSYSTEM_ROADMAP.yaml
```

If the roadmap cannot be fetched or read, fail closed for major architecture, shared-contract, cross-repository promotion, representation-format, or merge decisions. Read-only diagnosis is allowed.

## Mandatory ML maturity overlay

For any machine-learning, tensor/device/dtype, training, model I/O, compiler, data-pipeline, distributed, accelerator, serving, benchmark, or cross-repository ML work, also read:

`origin/agent/ecosystem-roadmap:.agent/ML_MATURITY_5_OF_5.yaml`

Recommended command:

```bash
git fetch origin agent/ecosystem-roadmap && \
git show origin/agent/ecosystem-roadmap:.agent/ML_MATURITY_5_OF_5.yaml
```

The ML maturity scores in that file are an audit baseline, not achievements. A dimension may be called `5/5` only after its listed end-to-end, interoperability, measurement, evidence, limitation, and exact-head CI exit criteria are actually satisfied. Never convert planned work, proxy benchmarks, compilation-only hardware support, or synthetic quality evidence into a higher maturity score.

## Mandatory reread points

Reread the roadmap and, for ML work, the ML maturity overlay:

1. at the start of every agent session;
2. before selecting the next major task;
3. before any cross-repository integration or promotion;
4. after any user instruction that changes ecosystem roles, invariants, strategy, or ML maturity priorities;
5. before opening or merging a PR that changes shared contracts, tensor representation, model/runtime behavior, evidence semantics, or public behavior.

## Ecosystem role

SciRust is the shared scientific, numerical, tensor, simulation, and reusable-computation substrate of the Memorithm ecosystem. It must not absorb product-specific responsibilities merely because another repository consumes SciRust.

In particular, preserve ownership boundaries with:

- `Memorithm/scirust-hub` — registry/orchestration/provenance control plane;
- `Memorithm/SciRust-Verify` — evidence dossier normalization and scoped verdicts;
- `Memorithm/SciCapsule` — user-facing portable capsule/trust/execution product;
- `Memorithm/forge` — execution-driven evolutionary candidate search;
- `Memorithm/ElasticXxx` — generic adaptive resource runtime;
- `Memorithm/NVIDIA-Native-Inference-Stack` — native NVIDIA model runtime;
- `Memorithm/FLAT-ATTENTION` — specialized attention engine;
- `Memorithm/SLHAv2` — compressed KV-cache product and real-model integration;
- `Memorithm/nonlocal-relativity-v2` — research incubator and promotion source for relativity work.

Cross-repository contracts are never assumed merely because code or concepts look similar. Read the other repository's bootstrap and current roadmap first.

## Core constraints

- correctness and declared semantics dominate optimization;
- no fabricated performance or scientific novelty;
- deterministic claims remain scoped to exact code path, target, features, toolchain, and evidence;
- optimized paths require an independent reference/oracle path;
- optional hardware backends remain optional unless an explicit versioned contract changes that policy;
- shared abstractions belong at the lowest correct reusable layer, not automatically in the largest repository;
- required CI must be green on the exact PR head before merge;
- missing roadmap, missing ML maturity overlay when applicable, missing provenance, or missing required evidence causes fail-closed behavior for major decisions.

## Mandatory roadmap maintenance

Update the off-main roadmap and the ML maturity overlay when applicable when:

- a roadmap or ML maturity phase changes status;
- a cross-repository contract is published, changed, or rejected;
- a shared ownership boundary changes;
- research work is promoted or rejected;
- an audited ML gap is closed, regresses, or is re-scoped;
- MSRV or major ecosystem compatibility policy changes.

Do not merge the roadmap or ML maturity overlay itself into the default branch unless the user explicitly requests it.

This file is only the bootstrap pointer. The off-main roadmap plus the ML maturity overlay are the persistent sources of current agent strategy and ML execution priorities.
