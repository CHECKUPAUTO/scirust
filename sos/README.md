# SOS — Scientific Operating System (implementation)

This is the **implementation workspace** for the Scientific Operating System.
The architecture it realizes is specified in [`docs/sos/`](../docs/sos/)
(RFC-0002); the discovery-loop subsystem is specified in
[`docs/sde/`](../docs/sde/) (RFC-0001).

SOS is a **separate Cargo workspace** from the SciRust workspace at the
repository root (RFC-0002 §11.6): it is excluded from the root workspace build,
has its own `Cargo.lock`, and consumes SciRust only from the two backend
adapter crates (`sos-scirust`, `sos-ccos`) — mechanically enforced by
[`sos/scripts/lint-deps.py`](scripts/lint-deps.py). This keeps SciRust's
"whole workspace builds on stable" gate intact and lets SOS evolve on its own
cadence.

## Status

Delivery is phased and **production-ready each phase** (RFC-0002 §12) — no
stubs, no TODOs, no placeholders cross a phase boundary.

| Phase | Scope | Status |
|-------|-------|--------|
| **P1 — Kernel & substrate** | `sos-core`, `sos-store`, `sos-provenance`, `sos-registry`, `sos-repro` (+ SOS CI) | **done** (`sos-repro` core landed on the merged scheduler — env-lock + drift + the level-aware reproduction contract; its backend-supplied `L2`/`L1` verdict is now real too, in `sos-scirust`'s `verdict` module — certificate-sum agreement for `L2`, a two-sample test at combined standard errors for `L1` — and `verify_rerun` joins the two halves: it pairs an original run's ledger against a re-run's, reads each node's declared level from the store, decides `L3`/`L0` by id and routes `L2`/`L1` to the numeric backend, refusing to verify across different plans or to assume a level it cannot read — and `verify_object` drives the whole thing from the store, finding the run that recorded an object, loading the `Plan` it names (a `Plan` is a storable, self-revalidating object now, so a stored plan tampered into a cycle fails to load rather than being re-executed), re-running it and verifying). The workspace's 4 dependency invariants are now CI-**enforced**, not just documented — see the dependency-invariant lint under Landed below |
| **P2 — Knowledge & Reasoning** | `sos-knowledge`, `sos-reasoning` | **done** (deterministic cores landed; Datalog / e-graph / theorem-proving deferred to `sos-scirust` per Invariant VIII) |
| **P3 — Discovery, Planning, Simulation** | `sos-workflow`, `sos-simulation`, `sos-planner`, re-homed `sde-*` stages | engine **cores landed** — the memoized scheduler, the backend-independent `Simulate` interface, and the planner (utility ranking + information-exhaustion + stopping rules). All three EIG-numerics tiers SDE §08 §3 names are now real, and `sos-planner`'s own contract is unchanged by any of them: the closed-form GP tier (`sos-scirust`'s `GpEigEstimator`), the Bayesian-optimization continuous-design-box search tier (`BoResult`/`search_best_design`, reusing `sos-planner`'s own `UtilityPolicy`), and the nested-Monte-Carlo discrete-hypothesis tier (`NestedMcEigEstimator`, with a real, computed standard error). `sos-simulation` now has three real backends — `sos-scirust`'s `Rk4OdeSimulator` (fixed-step ODE, `L3`), `Dopri5OdeSimulator` (adaptive ODE, `L2` with a real `CertifiedTrajectory` tolerance certificate), and `QuadratureSimulator` (adaptive Simpson quadrature, `L2` with a `CertifiedIntegral`) — with `sos-simulation`'s own `Simulate`/`Observation`/`Vcr` machinery unchanged by any of them. `sos-workflow`'s binding seam is closed too: `Dispatch` resolves a stage's plugin through `sos-registry` pinned to the content hash the stage recorded (a drifted implementation fails the run rather than silently computing something else), authorized against the study's capability grant. Manifest resolution has landed too: a TOML study file resolves to a validated `Plan` (`Manifest`/`resolve_manifest`), purely — no clock, no environment, no registry — so the same study memoizes identically on every machine, with unknown keys rejected rather than silently ignored. RFC-0002's `resolve(&manifest, &graph)` is half-done by design: inputs are content addresses, and naming them symbolically through the knowledge graph needs a query language this crate will not invent alone. The remaining solver numerics (nonlinear root-finding — a different fit than `Simulate`'s "observe an experiment" framing, so it needs its own look), that graph half, and the re-homed discovery stages still await `sos-scirust` / a frontend per Invariant VIII |
| **P4 — Curiosity & Theory** | `sos-curiosity`, `sos-theory` | **cores landed**, and `sos-curiosity`'s **information lens is now real**: `Curiosity::with_designs` consumes planner `Candidate`s carrying genuine EIG estimates and scores them by the same `CuriosityPolicy` as every structural question, so one agenda ranks "resolve this contradiction" against "run this experiment". This is the `sos-curiosity` → `sos-planner` composition edge — sanctioned by RFC-0002 §11.5 rule 3 and by the dependency lint since it landed, but unexercised until the EIG numerics existed. Analogy / Bayes-factor ranking / discriminating-experiment planning still await `scirust-graph` and the statistics backend per Invariant VIII |
| **P5 — Userland** | `sos-publication`, `sos-cli`, `sos-mcp` | `sos-publication` **core landed** — the publication is a verifiable projection of the object graph: content-addressed claims typed-bound to their evidence, the multi-phase claim/scope/reproducibility verifier, and deterministic Markdown/HTML/JSON. Re-execution of exhibits is `sos-workflow`'s job and real signing is `sos-provenance`'s per Invariant VIII; this crate consumes decisions, never recomputes them. `sos-cli` **porcelain landed** — `init`/`clone`/`push`/`log`/`know`/`ask`/`why`/`verify`/`diff`/`plan`/`publish`/`plugins` over the already-landed engines and the new persistent `FileStore`; `sos run` and a true `sos merge` are deferred, not stubbed. `sos run`'s two former blockers are both gone — a real `StageExecutor` backend exists in `sos-scirust`, and `sos-workflow` now resolves a TOML study to a `Plan` — but the CLI itself is not yet wired to them: it has no handler registry, so nothing in the binary can bind a study's plugins to code. `sos-mcp` **server landed** — the same syscalls as MCP tools over blocking stdio JSON-RPC (no async runtime, no new third-party dependency), with the untrusted-proposer tool opt-in per Invariant IX |
| **P6 — Backend adapters** | `sos-ccos` (cognitive), `sos-scirust` (computational) | `sos-ccos` **deterministic boundary landed** — the untrusted-proposer contract (Invariant IX): grounded, content-addressed proposals, the deterministic disposition gate, a tamper-evident attestation chain, and a no-LLM memory fallback. `sos-scirust` **all three gap-#1 EIG tiers, plus gap #3's first three backends, landed** — a closed-form GP-based EIG estimator wrapping `scirust-gp`; a Bayesian-optimization continuous-design-box search reusing `scirust-automl`'s seeded EI loop; a nested-Monte-Carlo estimator over finite `scirust-stats` discrete-hypothesis likelihoods, with a real, computed standard error rather than an asserted one; and three real `sos-simulation` `Simulate` backends wrapping `scirust-solvers` — fixed-step RK4 (`L3`, seedless-deterministic), adaptive Dormand-Prince 5(4) (`L2`, a real tolerance certificate carried in the output type since `Observation` has none of its own), and adaptive Simpson quadrature (`L2`, using the *strict* variant that errors on depth exhaustion rather than silently returning a non-compliant estimate). Four of the six capabilities are honestly tagged below `L3` (`L1`/`L1`/`L2`/`L2`, not `L3`) precisely where the underlying algorithm is seeded or tolerance-bounded, not by default; this is the sole crate CI-confirmed to touch `scirust-*`. (One of the quadrature backend's own tests caught a real bug along the way: the original fixed-point quantization scheme for hashing `f64` config fields collapsed distinct sub-nanoscale tolerances like `1e-10`/`1e-12` to the same encoded value — fixed workspace-wide by hashing configs as exact round-trip decimal strings instead of a quantized scale, still never a raw bit pattern.) Gap #2 is closed end to end: `sos-workflow`'s `Dispatch` supplies the registry-mediated binding mechanism, and `OdeStageHandler` is the first backend handler to use it — a hand-built `Plan` naming the ODE plugin now really runs RK4 under content-hash pinning and capability authorization, sealing its `Observation` into the object store as a content-addressed object the ledger references. Plans no longer have to be hand-constructed either — a TOML study resolves to one, and `sos-scirust`'s `manifest_to_result` suite runs a study written as text all the way to a stored trajectory. One path, precisely: only the RK4 backend is wired, no other engine's stages have handlers, and the CLI's `sos run` is still not connected to any of it. (That path caught a second content-addressing bug: `f64`s serialized as JSON *numbers* do not round-trip bit-exactly — 259 of 1539 trajectory floats came back changed — so trajectories are stored as exact decimal strings, the same representation the canonical encoding hashes.) Every other gap is deferred, not stubbed. The generative LLM/CCOS backend remains a deferred out-of-process backend per Invariant VIII |

### Landed

- **`sos-core`** — the kernel. The immutable, content-addressed
  [`Object`](sos-core/src/object.rs) envelope with deterministic canonical
  hashing, the honest four-level [`DeterminismLevel`](sos-core/src/determinism.rs)
  taxonomy, and full provenance / reproducibility metadata. Pure Rust, no FFI,
  `#![forbid(unsafe_code)]`, `#![deny(missing_docs)]`.
- **`sos-store`** — the Storage Layer (the kernel's filesystem). A
  content-addressed [`ObjectStore`](sos-store/src/store.rs) with typed
  [`put_object`/`get_object`](sos-store/src/store.rs) that **verify the content
  address on read and write**, content-addressed [`BlobRef`](sos-store/src/blob.rs)
  blobs, mutable named refs, and reachability [`gc`](sos-store/src/store.rs). Ships
  a complete in-memory backend ([`MemoryStore`](sos-store/src/mem.rs)) and a
  **persistent, filesystem-backed** [`FileStore`](sos-store/src/file.rs) — a
  git-style sharded-directory layout for objects/blobs, a small JSON ref index
  (never a caller-supplied name turned directly into a path — no
  path-traversal surface), atomic write-then-rename durability, and state that
  survives closing and reopening the store. A remote/object-storage backend
  implementing the same trait is a follow-on increment.
- **`sos-provenance`** — the Provenance Engine. A queryable
  [`ProvenanceGraph`](sos-provenance/src/graph.rs) over any store — `ancestors`
  ("why do we believe X"), `descendants` ("what breaks if X is retracted"),
  `roots`, `tips` — plus deterministic [environment capture](sos-provenance/src/env.rs)
  for the reproducibility key. (Signing is deferred to `sos-scirust`, keeping this
  crate backend-agnostic per Invariant VIII.)
- **`sos-registry`** — the Plugin System. Content-pinned
  [`PluginDescriptor`](sos-registry/src/descriptor.rs)s (name + version +
  content hash + [`Role`](sos-registry/src/descriptor.rs) + determinism level +
  capabilities + domains), a [`Registry`](sos-registry/src/registry.rs) that
  resolves by semantic version and **detects content-hash drift**, and
  least-privilege [capability authorization](sos-registry/src/capability.rs)
  (refuse-by-default).
- **`sos-knowledge`** — the Knowledge Engine (typed semantic graph). First-class
  relation [`Edge`](sos-knowledge/src/edge.rs)s (a typed
  [`Relation`](sos-knowledge/src/relation.rs) between two objects, sealed as
  content-addressed objects) and a deterministic
  [`KnowledgeGraph`](sos-knowledge/src/graph.rs) view with structural queries —
  `neighbors`, `in_neighbors`, `related`, shortest `path`. (Datalog / e-graph /
  analogy-by-isomorphism reasoning is deferred to `sos-reasoning` + `sos-scirust`
  per Invariant VIII.)
- **`sos-reasoning`** — the Reasoning Engine (deterministic, **LLM-free** core).
  Sound entailment over the knowledge graph — a directly-asserted edge, or a
  chain of a **transitive** relation — via [`Reason::entails`](sos-reasoning/src/reason.rs),
  returning a [`Conclusion`](sos-reasoning/src/reason.rs) whose
  [`Derivation`](sos-reasoning/src/derivation.rs) is itself a content-addressed,
  re-verifiable object that cites the exact edges used. Every result carries an
  honest [`Soundness`](sos-reasoning/src/soundness.rs) label (`Proof` vs a
  deterministic `Check`), and "not found" is `Undetermined`, never a false
  disproof. [`Reason::contradictions`](sos-reasoning/src/reason.rs) surfaces
  incompatibilities (asserted `contradicts` edges and mutual-`supersedes` cycles)
  as first-class [`Contradiction`](sos-reasoning/src/contradiction.rs) objects.
  (Datalog inference, SAT/SMT, e-graph saturation, theorem proving, and analogy
  by subgraph isomorphism are deferred to `sos-scirust` per Invariant VIII.)
- **`sos-curiosity`** — the Curiosity Engine (the OS **idle daemon**;
  deterministic, **LLM-free**). [`BeCurious::sweep`](sos-curiosity/src/sweep.rs)
  scans the knowledge graph and emits ranked
  [`ScientificQuestion`](sos-curiosity/src/question.rs)s, each a content-addressed
  object grounded in the real nodes it concerns and carrying a `Derivation`
  explaining *why* it is worth asking. Four deterministic lenses
  ([`Strategy`](sos-curiosity/src/strategy.rs)): **contradiction-hunt** (reusing
  `sos-reasoning`'s contradiction detection), **under-connected** (weakly-linked
  nodes), **weakly-supported** (claims refuted yet unsupported), and
  **maximal-information-gain** over planner-supplied designs
  ([`Curiosity::with_designs`](sos-curiosity/src/sweep.rs)). Scoring is an
  explicit, versioned [`CuriosityPolicy`](sos-curiosity/src/policy.rs) —
  **integer fixed-point, saturating** (bit-exact `L3` ranking, no opaque
  priorities, overflow-proof).

  The information lens is the `sos-curiosity` → `sos-planner` composition edge
  (RFC-0002 §11.5 rule 3). Curiosity never *computes* EIG — that stays the
  Planning Engine's job per Invariant VIII — it asks the question a real
  estimate justifies, scored by the *same* policy as every structural question
  so a contradiction and an experiment rank on one agenda. Two honesty
  properties are deliberate: it scores
  [`Estimate::lower_bound`](sos-planner/src/estimate.rs) (point − standard
  error), never the point estimate, so a noisy `0.9 ± 0.8` bit claim
  contributes what it can defend; and it **skips designs whose EIG is not
  significant** rather than emitting a question whose premise is "this might
  teach us nothing", mirroring the planner's own admission rule. Because EIG is
  unbounded but features must be bounded by `SCALE`, the saturation point is a
  caller-declared scale, not a magic constant.

  Wiring that lens exposed a latent bug in the default policy: its documented
  guarantee that a contradiction outranks everything else was spaced as
  `w_contradiction = 4` against `w_novelty + w_inv_cost = 3`, **excluding**
  `w_info_gain = 3` — sound only while `Features::info_gain` was hardcoded to
  `0`. Once the lens made that field live, the non-contradiction ceiling became
  `6·SCALE` and a sufficiently attractive design could displace an unresolved
  contradiction from the top of the agenda. `w_contradiction` is now `7`, so
  the dominance holds for *every* feature vector rather than accidentally; a
  test pins the arithmetic. (This changes `CuriosityPolicy::default()`'s
  content hash — acceptable pre-1.0, and preferable to the default agenda
  silently changing behavior.) Cross-domain analogy via `scirust-graph` and
  cognitive proposals via `sos-ccos` remain deferred per Invariant VIII.
- **`sos-theory`** — the Theory Engine (deterministic). Theories are
  **first-class, immutable, evolving** objects: a
  [`Theory`](sos-theory/src/theory.rs) records all ten mandate fields (axioms,
  assumptions, equations, [`Scope`](sos-theory/src/scope.rs) domain of validity,
  supporting **and** contradicting evidence, confidence, citations, revision
  parent, competitors) as ids into the graph — a view over provenance, not a
  document. [`Theory::revise`](sos-theory/src/theory.rs) evolves a theory into a
  *new* node that **retains its anomalies** (contradicting evidence is never
  hidden) and links its parent; the [`Theories`](sos-theory/src/engine.rs) engine
  walks the full [`revision_chain`](sos-theory/src/engine.rs) (old theories stay
  queryable) and [`compare`](sos-theory/src/engine.rs)s rivals over their shared
  domain, so competitors coexist rather than being forced to a single winner.
  (Bayes-factor `Confidence` ranking and discriminating-experiment planning are
  deferred to the statistics backend + `sos-planner` per Invariant VIII.)
- **`sos-workflow`** — the Workflow Engine (the OS **scheduler**; a *build system
  for knowledge*). An immutable [`Plan`](sos-workflow/src/plan.rs) DAG of
  [`Stage`](sos-workflow/src/plan.rs)s with a **deterministic** topological
  schedule (ties by `StageId`); the content-addressed
  [`CacheKey`](sos-workflow/src/cache.rs) — `hash(descriptor ⊕ inputs ⊕ config ⊕
  seed ⊕ env)` — that gives **reproducibility and incremental compute from one
  mechanism**; and [`run_plan`](sos-workflow/src/engine.rs), the memoized driver
  (cache-hit ⇒ reuse, cache-miss ⇒ execute via a pluggable
  [`StageExecutor`](sos-workflow/src/engine.rs)) that records the schedule taken
  in a content-addressed [`RunLedger`](sos-workflow/src/ledger.rs). Re-running an
  unchanged plan against a warm [`Memo`](sos-workflow/src/engine.rs) is all cache
  hits — provably identical, nearly free, and the property that makes a crashed
  run resumable.

  [`Dispatch`](sos-workflow/src/dispatch.rs) closes the binding seam: the
  scheduler could always order and memoize stages, but nothing turned a stage's
  plugin *name* into running code. `Dispatch` resolves each stage through
  [`sos-registry`](sos-registry/src/registry.rs) **pinned to the content hash
  the stage recorded**, so an implementation that drifted under the same name
  and version fails the run rather than silently computing something else and
  calling it the same result — the RFC's drift guarantee, now enforced rather
  than asserted. Every binding is authorized against the study's capability
  `Grant`, refusing by default, so a plugin gets the GPU or the right to cause
  effects because the study granted them, not because it asked; and a
  descriptor that resolves with no registered handler is an error, never a
  quietly-empty output. `Dispatch` is itself a `StageExecutor`, so it drops
  into `run_plan` with memoization and the ledger above it unchanged — and
  because cache hits never call the executor, binding and capability checks
  fall only on work that actually runs. (Stage *logic* itself, manifest
  resolution, and stopping rules remain with the engine plugins /
  `sos-scirust` / `sos-planner` per Invariant VIII — no stub.)
- **`sos-repro`** — the Reproducibility Engine (the *Nix analogy*). Where
  provenance *records* the environment, this **pins and re-realizes** it: an
  [`EnvLock`](sos-repro/src/lock.rs) is the hermetic lockfile (toolchain, backend
  versions + content hashes, hardware, OS) whose `env_digest` keys the workflow
  cache, plus itemized [`Drift`](sos-repro/src/lock.rs) detection — "binds the
  same pins or **declares** the drift". The level-aware **reproduction contract**
  ([`verify_reproduction`](sos-repro/src/contract.rs)) decides `L3` bit-exact and
  `L0` replay by object-id equality and localizes any deviation to a node and its
  level; `L2` within-certificate / `L1` in-distribution take a backend-supplied
  verdict. [`rerun`](sos-repro/src/rerun.rs) re-realizes a `sos-workflow` plan
  under a pinned lock — a binding lock reproduces from cache, a drifted lock
  recomputes. (The numeric/statistical `L2`/`L1` evaluation and a store-driven
  `verify(object)` that walks + re-executes a sub-DAG are deferred to
  `sos-scirust` per Invariant VIII — no stub.)
- **`sos-simulation`** — the Simulation Engine (backend-independent core). A
  simulation is *an experiment whose executor is a solver*: the
  [`Simulate`](sos-simulation/src/simulate.rs) trait is the syscall the discovery
  loop names instead of a concrete backend, so the loop is identical whether
  evidence comes from a PDE solve or a wet lab (solvers are `sos-scirust`
  backends implementing the trait — **no solver here**). Every result is an
  [`Observation`](sos-simulation/src/observation.rs) that **stamps the honest
  [`DeterminismLevel`](sos-core/src/determinism.rs)** the backend realized (`L3`
  bit-exact … `L1` seeded-stochastic), so nothing is presented as more
  reproducible than its backend allows. A record/replay
  [`Vcr`](sos-simulation/src/vcr.rs) memoizes runs — perform a simulation once,
  replay it identically thereafter — letting an expensive or one-shot simulation
  live inside a reproducible workflow. (The capability-gated world-effect boundary
  is the Workflow executor seam's job per Invariant VIII.)
- **`sos-planner`** — the Planning Engine (deterministic; the engine `sos-curiosity`
  and `sos-theory` defer their information-gain queries to). It turns
  expected-information-gain estimates into the decision *"run this experiment
  next"* — or the honest *"information is exhausted, stop"*. An
  [`Estimate`](sos-planner/src/estimate.rs) **carries its own uncertainty** (point,
  standard error, level), a versioned [`UtilityPolicy`](sos-planner/src/policy.rs)
  (`U = EIG/cost`) scores it against a [`Cost`](sos-planner/src/estimate.rs) model,
  and the myopic [`GreedyPlanner`](sos-planner/src/planner.rs) ranks candidates
  into a content-addressed, defensible [`Plan`](sos-planner/src/plan.rs) —
  recommending `ξ*` or [`InformationExhausted`](sos-planner/src/plan.rs) when no
  design clears the `eig < ε` floor. Composable
  [`StoppingRule`](sos-planner/src/stopping.rs)s (`posterior_mass > p`, `eig < ε`,
  `budget_exhausted`) close the discovery loop. Integer fixed-point (EIG in
  millibits) — no opaque score. (**Computing** EIG — GP predictive variance,
  nested Monte-Carlo — is `sos-scirust`'s job per Invariant VIII; this crate
  consumes estimates.)
- **`sos-publication`** — the Publication & Claim-Verification Engine (the
  *reproducibility crisis, inverted*). A **publication is a verifiable
  projection of the object graph**: a content-addressed
  [`Publication`](sos-publication/src/publication.rs) whose first-class
  [`Claim`](sos-publication/src/claim.rs)s are wired to the graph by typed
  [`ClaimBinding`](sos-publication/src/claim.rs)s (directly/indirectly supports,
  **contradicts**, supplies-method/data, reproduces, …), so *which object
  supports this claim?* has a mechanical answer. The multi-phase
  [`verify`](sos-publication/src/verify.rs) resolves every dependency, takes the
  declared-scope [closure](sos-publication/src/source.rs), and assigns each claim
  a categorical [`ClaimStatus`](sos-publication/src/verify.rs)
  (supported / partially / **contradicted** / unresolved / unverifiable /
  dependency-missing / reproducibility-failed / policy-rejected) under a
  versioned, non-opaque [`StandardPolicy`](sos-publication/src/policy.rs) —
  integrity is never mistaken for truth, and a contradiction is reported, never
  hidden. Plus [`verify_exhibits`](sos-publication/src/verify.rs) (figure/table
  drift localization), [`check_release`](sos-publication/src/verify.rs)
  ("changed since release?"), a semantic [`diff`](sos-publication/src/diff.rs),
  and deterministic Markdown/HTML/JSON
  [`render`](sos-publication/src/render.rs). (Re-executing exhibits is
  `sos-workflow`'s job and real Merkle/Lamport signing is `sos-provenance`'s per
  Invariant VIII; LaTeX/PDF need a typesetting backend — no stub.)
- **`sos-ccos`** — the Cognitive Backend Adapter (the *untrusted proposer*). The
  one place SOS touches generative intelligence, and the one place it must never
  trust: **cognition proposes, determinism disposes** (Invariant IX). A
  [`Proposal`](sos-ccos/src/proposal.rs) is a content-addressed, **untrusted**
  suggestion that must **ground** in real objects; [`dispose`](sos-ccos/src/disposition.rs)
  turns a deterministic engine's ruling into an [`Admission`](sos-ccos/src/disposition.rs),
  and the gate is not bypassable — a tampered or ungrounded proposal is rejected
  even under an `Admit`, and a [`Trusted`](sos-ccos/src/disposition.rs) reference
  exists *only* via an admitted admission (Invariant IX enforced in the type
  system). Every cognitive act is recorded in a tamper-evident
  [`CcosChain`](sos-ccos/src/attest.rs) (`input→output→chain` hashes, `verify()`
  localizes any alteration), acts are capability-scoped
  ([`Cognition`](sos-ccos/src/session.rs), refuse-by-default), and
  [`LocalMemory`](sos-ccos/src/memory.rs) is a deterministic no-LLM fallback
  (recall degrades to exact structural overlap). (The generative LLM/CCOS backend
  and embedding-backed recall are the deferred out-of-process backend per
  Invariant VIII — no stub, no fake cognition.)
- **`sos-scirust`** — the Computational Backend Adapter, and the only other
  crate the dependency lint permits to name `scirust-*` (Invariant VIII). All
  three of gap #1's EIG tiers are landed, and `sos-planner`'s
  ranking/stopping-rule machinery is **unchanged** by any of them: each tier
  only produces the same `Estimate`/`Candidate` types a consumer always could,
  now backed by real numerics instead of a hand-supplied number.
  [`GpEigEstimator`](sos-scirust/src/eig.rs) (**tier 1**, closed-form) wraps
  [`scirust-gp`](../scirust-gp)'s exact Gaussian-process posterior variance in
  the closed-form Gaussian-channel mutual-information formula
  (`0.5·log2(1 + var/noise)` bits) — `L3`, zero standard error, since the
  formula is analytic in the GP's own variance, not sampled.
  [`BoResult`/`search_best_design`](sos-scirust/src/bo.rs) (**tier 2**,
  search) answers a different question — the best design in a whole
  *continuous* box, not a ranking of a pre-enumerated set — by reusing
  `scirust-automl`'s seeded `bayesian_optimize`/`expected_improvement` loop to
  maximize `sos-planner`'s own
  [`UtilityPolicy::utility`](sos-planner/src/policy.rs) directly (no duplicate
  scalarization); `L1`, honestly, since *which* point a seeded search returns
  is a function of the seed even though the EIG value at that point is itself
  exact (SDE §08 §6's `automl` classification).
  [`NestedMcEigEstimator`](sos-scirust/src/nmc.rs) (**tier 3**, nested Monte
  Carlo) is for a third, genuinely different scenario — discrete hypothesis
  discrimination with non-Gaussian likelihoods, a finite set of
  `scirust-stats` `DiscreteDistribution`s (`Poisson`, `Binomial`, ...), one per
  hypothesis: the inner Bayes update is exact (a finite, `K`-term
  log-sum-exp), only the outer expectation over the observation is seeded
  Monte Carlo (`scirust-stats`' `SplitMix64`); `L1`, and — unlike tiers 1/2 —
  a genuinely non-zero standard error, computed from the real spread of the
  Monte-Carlo draws
  ([`scirust_stats::describe::std_error`](../scirust-stats/src/describe.rs)),
  never asserted.

  Gap #3 (`sos-simulation` backends) now has three entries —
  [`sos-scirust/src/ode.rs`](sos-scirust/src/ode.rs)'s `Rk4OdeSimulator` and
  `Dopri5OdeSimulator`, and
  [`sos-scirust/src/quadrature.rs`](sos-scirust/src/quadrature.rs)'s
  `QuadratureSimulator` — spanning both determinism levels SDE §08 §2 names
  for this family. `sos-simulation` ships the backend-independent
  [`Simulate`](sos-simulation/src/simulate.rs) syscall, `Observation`'s honest
  determinism stamping, and the `Vcr` record/replay memo, but (like
  `sos-planner`) implements no solver itself. `Rk4OdeSimulator` integrates
  `dy/dt = f(t, y)` via `scirust-solvers`' fixed-step RK4 and is `L3` —
  RFC-0002 §08 §1 classifies `scirust-solvers` itself as
  *seedless-deterministic*, and the fixed-step loop bears that out concretely
  (a fixed sequence of scalar `f64` operations, no adaptive branching).
  `Dopri5OdeSimulator` integrates the same equation via adaptive
  Dormand-Prince 5(4) (the algorithm behind `scipy.integrate.RK45` / MATLAB's
  `ode45`) and is `L2`, not `L3`: every step's accept/reject decision branches
  on a computed error norm against the caller's `rtol`/`atol`, the textbook
  "iterative solver to a tolerance" case. Because `Observation` has no
  dedicated certificate field, the certificate lives in the output type
  itself — `CertifiedTrajectory` carries the trajectory *and* the
  `rtol`/`atol`/accepted/rejected-step bookkeeping that bounds its accuracy.
  `QuadratureSimulator` estimates `∫ₐᵇ f(x) dx` via adaptive Simpson
  quadrature — also `L2`, same reasoning — using the *strict* variant that
  errors on recursion-depth exhaustion rather than silently returning a
  non-compliant estimate (a certificate that could be wrong isn't one); its
  `CertifiedIntegral` needs no accepted/rejected bookkeeping, since a
  *successful* strict call already guarantees the declared tolerance was met.
  `SimDescriptor` is caller-supplied rather than hardcoded for all three,
  since two different models sharing the same integrator/quadrature code need
  distinct descriptors to cache distinctly.

  All three configs carry `f64` fields, and the kernel's `CanonicalEncoder`
  is deliberately float-free (`sos_core::canonical` module docs) — but
  they're encoded *exactly*, as each value's shortest round-trip decimal
  string, rather than quantized to a fixed-point scale. That's a correction:
  the quadrature backend's own test suite caught a real bug where the
  original nanoscale (`1e9`) fixed-point quantization collapsed distinct
  sub-nanoscale tolerances (`1e-10` vs. `1e-12`, both realistic solver
  tolerances) to the same encoded value — a genuine cache-key collision. The
  exact-string encoding (`sos-scirust/src/solver.rs`, shared by `ode` and
  `quadrature`) has no scale to choose and so nothing to collide, at any
  magnitude, while still never hashing a raw bit pattern; `-0.0` is
  normalized to `0.0` first so the two, which compare `==`, also encode
  identically. This affects content-addressing only — every integration
  always runs at full `f64` precision regardless.

  [`stage`](sos-scirust/src/stage.rs) wires the first of those backends
  through `sos-workflow`'s `Dispatch` as a real handler, which makes one
  execution path work end to end: a hand-built `Plan` whose stage names the
  ODE plugin now runs `scirust-solvers`' RK4 under real registry pinning and
  real capability authorization, and its `Observation` is sealed into the
  object store as a content-addressed `OdeTrajectory` object whose id the
  ledger records. Stated precisely, because it is one path and not a general
  capability: only `Rk4OdeSimulator` has a handler (`Dopri5OdeSimulator` and
  `QuadratureSimulator` do not), plans are still hand-constructed since
  and no other engine's stages are wired. Plans can now come from a TOML study rather than only from code (`sos-workflow`'s `Manifest`), which `tests/manifest_to_result.rs` exercises end to end.

  That path also surfaced a storage bug worth recording, of the same family
  as the quantization one above. Serializing a trajectory's `f64`s as JSON
  *numbers* is not bit-exact: over one 513-step run, **259 of 1539 floats
  came back from `serde_json` with different bits** (e.g.
  `0.009203884727313847` → `0.009203884727313849`), so the reloaded object
  hashed to a different id and the store correctly rejected it as corrupt.
  Trajectory floats are therefore stored as shortest round-trip decimal
  strings and read back with `str::parse`, which loses nothing on the same
  1539 values — and which makes the serialized form and the hashed canonical
  form the same text, so the two cannot drift apart.

  Every other gap in the `sos-scirust` scoping plan is deferred, not stubbed:
  gap #3 itself still has `scirust-solvers`' nonlinear (Newton/Broyden)
  root-finding (a different fit than `Simulate`'s "observe an experiment"
  framing, so it needs its own look rather than a mechanical repeat of this
  pattern), plus `scirust-signal`/`scirust-sim`'s executor kinds, ahead of
  it.
- **`sos-cli`** — the `sos` command-line porcelain (RFC-0002 §10.4), the
  first-ever user-facing entry point into SOS. A thin, git-shaped shell adding
  no new compute of its own: `sos init`/`clone`/`push` manage a reasoning
  repository (a [`FileStore`](sos-store/src/file.rs)); `sos log` lists every
  object; `sos know` queries the knowledge graph (neighbors / related / path);
  `sos ask` runs a curiosity sweep; `sos why` prints the provenance behind an
  object; `sos verify` checks structural identity plus, for every `Body` kind
  landed so far, a real recomputed content-hash check; `sos diff` compares two
  studies' ancestor sets (reusing `sos-publication`'s dependency-closure
  walker); `sos plan`/`sos publish`/`sos plugins` consume already-computed
  candidates/publications/descriptors exactly as their engines are scoped to.
  Hand-rolled argument parsing (`args.rs`) — no new third-party dependency,
  matching `scirust-cli`'s own convention. (`sos run` is still deferred, but for a
  narrower reason than before: both of its blockers are gone — a real
  [`StageExecutor`](sos-workflow/src/engine.rs) backend exists
  (`sos-scirust`'s [`OdeStageHandler`](sos-scirust/src/stage.rs)) and
  [`Manifest`](sos-workflow/src/manifest.rs) turns a study file into a `Plan`
  — and what remains is wiring this binary to them: the CLI has no handler
  registry, so it cannot bind a study's plugin names to code. A true
  `sos merge` needs conflict-resolution semantics
  no crate has designed yet; neither is stubbed. A network remote for
  `clone`/`push` is `sos-mcp`'s domain.)
- **`sos-mcp`** — the SOS Model Context Protocol server. Exposes the same
  syscalls as `sos-cli` (`sos_log`/`sos_why`/`sos_verify`/`sos_diff`/`sos_know`/
  `sos_ask`/`sos_plan`/`sos_publish`/`sos_plugins`) as MCP tools over a
  blocking stdio JSON-RPC 2.0 transport
  ([`server.rs`](sos-mcp/src/server.rs)) — no async runtime, no third-party
  MCP/JSON-RPC crate, mirroring `scirust-mcp`'s own proven pattern in this
  repository rather than adding a new dependency. Read tools call
  [`sos-cli`](sos-cli)'s command functions directly (one implementation, not
  two); `sos_plan`/`sos_publish`/`sos_plugins` take their data **inline** as
  MCP arguments rather than a file path. Every tool call is attested into a
  [`CcosChain`](sos-ccos/src/attest.rs) — reusing `sos-ccos`'s tamper-evident
  chain rather than a bespoke audit log, because an MCP call from an agent
  *is* a cognitive act in RFC-0002's framing. `sos_propose` — the
  untrusted-proposer entry point (Invariant IX) — is **opt-in**: a
  [`RegistryProfile::Query`](sos-mcp/src/lib.rs) server never registers it at
  all, so a strictly read-only deployment is one by construction, not by
  policy alone.
- **Dependency-invariant lint** — [`sos/scripts/lint-deps.py`](scripts/lint-deps.py)
  reads real `cargo metadata` and mechanically enforces all four rules from
  [11 §5](../docs/sos/11-workspace-and-crate-graph.md#5-dependency-invariants-enforced-in-ci):
  `sos-core` is the universal sink (every crate reaches it, directly or
  transitively), the workspace graph is acyclic, engine-to-engine composition
  is confined to the documented edges, and `scirust-*`/CCOS naming is confined
  to `sos-scirust`/`sos-ccos` (Invariant VIII) — in *any* dependency kind, so a
  test-only leak fails it too. Runs as its own job
  ([`sos-lint-deps`](../.github/workflows/sos-ci.yml)) on every `sos/` change.

## Engineering standards (the gate)

Every crate must pass, on every change:

```sh
cargo fmt   --manifest-path sos/Cargo.toml --all --check
cargo clippy --manifest-path sos/Cargo.toml --all-targets -- -D warnings
cargo test  --manifest-path sos/Cargo.toml
python3 sos/scripts/lint-deps.py
```

- Rust **stable**, MSRV **1.89**.
- 100 % documented public API (`#![deny(missing_docs)]`).
- Deterministic + property-based tests (seeded generators; no unseeded
  randomness, no wall-clock in hashed state).
- No `unsafe` (`#![forbid(unsafe_code)]`), no FFI.
- The 4 dependency invariants ([11 §5](../docs/sos/11-workspace-and-crate-graph.md#5-dependency-invariants-enforced-in-ci))
  hold, mechanically — [`sos/scripts/lint-deps.py`](scripts/lint-deps.py) reads
  real `cargo metadata`, not intent.

> SOS is a separate, excluded workspace, so the repository's root CI does not
> build it. A dedicated **SOS CI** workflow
> ([`.github/workflows/sos-ci.yml`](../.github/workflows/sos-ci.yml)) gates it
> upstream with the commands above — fmt (on the repo's pinned nightly, since
> `rustfmt.toml` uses unstable options), clippy `-D warnings`, `test` on stable,
> an MSRV 1.89 check, and the dependency-invariant lint — path-filtered to run
> only when `sos/` changes. The workspace's `Cargo.lock` is committed so CI
> builds with `--locked` are reproducible.
