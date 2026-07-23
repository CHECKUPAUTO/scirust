# 08 · Workflow & Simulation

> [← Discovery, Experiment & Theory](./07-discovery-experiment-theory.md) · [Provenance, Reproducibility & Storage →](./09-provenance-reproducibility-storage.md)

The Workflow Engine is the **scheduler** of the scientific OS; the Simulation
Engine is one class of executor it schedules. The workflow mechanics are
specified in [SDE 04](../sde/04-workflow-engine.md) and inherited; this chapter
states the SOS generalization and the backend-independent simulation interface.

Rust is illustrative sketch.

---

## 1. The workflow model: immutable, memoized DAGs

A workflow is an immutable `Object<Workflow>` — a DAG of engine invocations, each
node a pure pass over SOS-IR (except the effectful simulation/execution
boundary). The model is exactly the SDE workflow engine's, with four properties
the mandate restates:

| Mandate requirement | Mechanism ([SDE 04](../sde/04-workflow-engine.md)) |
|---|---|
| every node reproducible | pure stages + captured `ReproMeta` + determinism levels |
| every edge explicit | provenance edges are the DAG; no implicit data flow |
| every execution memoized | content-addressed `cache_key` per stage invocation |
| result cached by content hash | idempotent object store `put`; identical inputs ⇒ cache hit |
| incremental recomputation | change anything in a `cache_key` and only the affected sub-DAG re-runs |

```rust
pub trait Workflow {                      // the scheduler "syscall"
    fn resolve(&self, manifest: &Manifest, graph: &Graph) -> PlanGraph;
    fn run(&self, plan: &PlanGraph) -> RunLedger;   // records the schedule taken
}
```

---

## 2. Content-addressed memoization

Restated because it is the property that unifies reproducibility and incremental
compute (one mechanism, both guarantees — [SDE 04 §3](../sde/04-workflow-engine.md#3-the-scheduler-content-addressed-memoized-incremental)):

```
cache_key = hash( stage.descriptor      // engine/plugin name + version + content hash
                ⊕ input_object_ids       // exact nodes consumed
                ⊕ config_hash
                ⊕ seed                    // mandatory
                ⊕ env_digest )            // toolchain / backend / hardware class
```

If `cache_key` is already in the store, the stage does not run — its outputs are
returned from cache, and re-running an unchanged workflow is provably identical
and nearly free. This is what makes SOS a *build system for knowledge*: a solved
sub-DAG is never re-solved.

---

## 3. Generalization: the scheduler runs all engines

The SDE workflow engine scheduled the ten discovery stages. `sos-workflow`
schedules **any** engine invocation — a curiosity sweep, a reasoning derivation, a
knowledge assertion, a theory revision, a publication render — because all of them
are the same shape: a pure (or capability-gated) pass that reads objects and
appends objects. The scheduler is engine-agnostic; it sees `StageDescriptor`s and
`ObjectId`s, not "discovery" vs. "curiosity." This is why the OS metaphor holds:
one scheduler, many processes.

The scheduler is deterministic (topological order, ties broken by `ObjectId`), so
even the *schedule* is reproducible and recorded in the `RunLedger`
([02 §3](./02-system-architecture.md#3-data-plane-vs-control-plane)). Parallel
execution is sound because stages are pure; results never depend on execution
order.

---

## 4. The effect boundary (unchanged, restated)

The only impure nodes are `Execution` and `Observation` — the seam where a
workflow touches a simulator, instrument, market, or human. Effects go through a
signed, time-boxed `Capability` and are **recorded** so replay reuses the
recording ([SDE 04 §5](../sde/04-workflow-engine.md#5-the-effect-boundary-executors)).
A replayed workflow is L3 from the observations down, even when the original
experiment was a one-shot physical event. Simulations are the L2/L3 in-process
case of this boundary.

---

## 5. The Simulation Engine

A **simulation is an experiment whose executor is a solver** ([03 §4](./03-object-model.md#4-two-clarifying-distinctions)):
it inherits pre-registration, cost, provenance, memoization, and the determinism
taxonomy. The Simulation Engine's job is to present a **backend-independent
interface** so the Discovery loop is identical whether evidence comes from a wet
lab or a PDE solve.

```rust
pub trait Simulate {                      // a "syscall"; SciRust is the default impl
    type Config;   type Output;
    fn run(&self, cfg: &Self::Config, seed: u64, cap: &Capability)
        -> Result<Observation<Self::Output>, SimError>;
    fn level(&self) -> DeterminismLevel;  // L3 bit-exact ... L1 seeded-stochastic
}
```

Backend-independence is real, not nominal: `Simulate` lives in `sos-core`;
`sos-scirust` provides the default implementations; a different backend (an
external HPC solver over MCP, a WASM-sandboxed model) implements the same trait
and declares its own determinism level. The Discovery Engine never names a
backend — it names a `Simulate` capability resolved through the registry.

### Simulation domains → SciRust backends

The mandate's simulation domains each map to existing SciRust crates (the fuller
domain map is [SDE 08 §4](../sde/08-scirust-integration.md#4-domain-map-which-scirust-crates-power-which-field)):

| Simulation domain | Primary SciRust backend | Determinism |
|---|---|---|
| Signal processing | `scirust-signal` (FFT, filters, CFAR, DoA) | L3 |
| Optimization | `scirust-solvers` (`optimize`: BFGS, Nelder-Mead, SPG), `scirust-automl` | L2/L1 |
| ODE | `scirust-solvers` (`ode`: RK4, Dormand-Prince 5(4)), `scirust-stiff` (stiff) | L2 + tolerance certificate |
| PDE | method-of-lines over `scirust-solvers`/`scirust-stiff`; `scirust-fluids` (CFD); `scirust-fractional` | L2 |
| Symbolic mathematics | `scirust-symbolic`, `scirust-modalg` (exact) | L3 |
| Machine learning | `scirust-learning`, `scirust-automl`, `scirust-nas`, `scirust-autodiff` | L1 (seeded) |
| Quantum simulation | `scirust-tn` (tensor networks); quantum track (`docs/research/SCIRUST_QUANTUM_ROADMAP`) | L2/L3 |
| Robotics | `scirust-robotics`, `scirust-nav`, `scirust-estimation`, `scirust-control` | L2 |
| Finance | `scirust-trader`, `scirust-forecast`, `scirust-seasonal` | L1 (record/replay ticks) |
| Biology | `scirust-biomed` | L0/L2 |
| Chemistry | `scirust-thermo`, `scirust-units`, `scirust-fluids` | L2 |
| Engineering | `scirust-control`, `scirust-fatigue`, `scirust-electrotech`, `scirust-civil`, `scirust-reliability` | L2 |

Each simulation run is a memoized workflow node: re-running an identical
simulation (same config, seed, backend version, hardware class) is a cache hit and
returns the byte-identical prior `Observation` — reproducibility and compute
savings from the same mechanism.

---

## 6. Determinism of simulation, honestly

Simulations sit across the determinism taxonomy, and SOS records which:

- **L3** — symbolic/exact-algebra simulations (`scirust-symbolic`,
  `scirust-modalg`) and fixed-order integer/`CpuBackend` numerics.
- **L2** — floating-point solvers to a tolerance; the tolerance is the
  `Certificate` (`bound_ulps`), and cross-hardware results reproduce *within ε*,
  not to the bit ([09 §4](./09-provenance-reproducibility-storage.md#4-the-determinism-taxonomy-carried-through)).
- **L1** — seeded stochastic simulations (ML training, Monte-Carlo); reproduce
  *in distribution* given the recorded seed.

The level is declared by the backend, propagated by the scheduler, and stamped on
every `Observation` — so a study built on an L1 simulation says so, and its
reproducibility contract reflects it. No simulation is presented as more
reproducible than its backend allows.

---

> [← Discovery, Experiment & Theory](./07-discovery-experiment-theory.md) · [Provenance, Reproducibility & Storage →](./09-provenance-reproducibility-storage.md)
