# 10 · Roadmap, Risks & Future Research

> [← Workspace & Crates](./09-workspace-and-crates.md) · [Back to README](./README.md)

---

## Part A · Roadmap

The roadmap is **vertical-slice first**: get one real question flowing end-to-end
through a thin version of every stage before widening any single stage. The first
thing SDE reproduces is SciRust's *own* research history — the strongest possible
dogfooding signal ([README](./README.md#why-this-repo-specifically)).

```mermaid
flowchart LR
    M0["M0 · Spec<br/>(this RFC)"] --> M1["M1 · Vertical slice<br/>+ dogfood docs/kb"]
    M1 --> M2["M2 · Planner<br/>(EIG loop closes)"]
    M2 --> M3["M3 · Domains<br/>(3+ fields)"]
    M3 --> M4["M4 · Federation<br/>+ executable papers"]
    M4 --> M5["M5 · Ecosystem<br/>(WASM/MCP plugins)"]
```

### M0 — Specification & object model *(this document)*
- **Deliverable:** RFC-0001 (this directory); frozen `Object<B>` envelope,
  determinism taxonomy, stage-trait signatures, hashing spec.
- **Exit:** the substrate ABI in `sde-core` is agreed; schema v1 is pinned.

### M1 — Vertical slice (walking skeleton)
- **Deliverable:** `sde-core`, `sde-store`, `sde-provenance`, `sde-workflow` +
  one *thin* plugin per stage in `sde-scirust`, driven by `sde-cli run`.
- **Proof:** re-express one existing `docs/kb/` run (`topic_init →
  problem_decompose → hypothesis_gen → … → synthesis`) as a hashed, replayable
  DAG; `sde verify` is green; a second machine reproduces identical L3 IDs.
- **Why first:** it exercises every seam (objects, hashing, provenance, effect
  boundary, memoization) at minimum width, surfacing integration risk early.

### M2 — Close the loop with the planner
- **Deliverable:** `sde-infotheory` (closed-form GP EIG + nested-MC fallback),
  `sde-planner` (EIG/cost, stopping rules), iteration controller.
- **Proof:** on a synthetic law-discovery task with a known answer, the planner's
  recommended experiments reach the `posterior_mass > 0.99` stop in
  *measurably fewer* experiments than random/grid design — and the graph shows
  the bits bought at each step.

### M3 — Breadth across domains
- **Deliverable:** three `Domain` impls from different quadrants of the brief —
  e.g. **optimization** (`scirust-solvers`/`automl`), **signal/engineering**
  (`scirust-signal`), and **ML/AutoML** (`scirust-automl`/`nas`).
- **Proof:** the same engine, planner, and provenance run all three unmodified;
  a new domain is demonstrably "implement one trait," measured in reviewer-hours.

### M4 — Federation & executable papers
- **Deliverable:** `sde-report` (executable paper: every figure re-executes from
  its node, publication root signed); graph `merge` with explicit conflict
  objects; `sde clone`/`push` over the object store.
- **Proof:** two independently-run studies of one question merge into a single
  DAG; a rendered paper's figures regenerate byte-for-byte at L3 from `sde
  report --verify`.

### M5 — Ecosystem & polyglot backends
- **Deliverable:** `sde-registry` WASM + `sde-mcp` transports GA; a `sde-python`
  reference adapter; a plugin index (`sde plugins find`).
- **Proof:** a non-Rust backend and a sandboxed community plugin run in a real
  study with correctly-declared determinism levels; `scirust-sciagent` drives a
  full loop over MCP with every action recorded.

### Stability milestone — 1.0
`sde-core`'s ABI and the on-disk/wire formats commit to semver stability;
schema migrations become RFC-gated and append-only. Only after M4's
reproducibility contract holds across machines and backends.

---

## Part B · Risks & mitigations

Stated plainly, worst-first. Each is a real threat to the thesis, with a
concrete mitigation and an honest residual.

### R1 — Cross-hardware floating-point non-reproducibility *(highest)*
- **Threat:** the same numeric workflow yields different bits on different CPUs,
  BLAS backends, or GPUs, undermining "identical outputs."
- **Mitigation:** the **determinism taxonomy is the mitigation** — L3 is reserved
  for integer/symbolic/fixed-reduction-order code; numeric results are L2 with a
  certified tolerance ([06 §3](./06-provenance-and-reproducibility.md#3-determinism-levels-propagated-and-certified)).
  SDE builds on `scirust-gpu`'s bit-exact `CpuBackend` oracle + Kahan summation
  for a portable L3 reference, and validates other backends against it.
- **Residual:** L2 studies reproduce *within ε*, not to the bit. This is
  disclosed per-node, never hidden — the design's core honesty move.

### R2 — Storage growth from "nothing is ever lost"
- **Threat:** total retention balloons.
- **Mitigation:** content-addressed dedup (shared sub-DAGs stored once), large
  arrays as side blobs, opt-in reachability GC ([06 §7](./06-provenance-and-reproducibility.md#7-what-nothing-is-ever-lost-costs-and-how-it-is-bounded)).
- **Residual:** long-lived shared stores still need capacity planning; the store
  backend is pluggable (S3-class) precisely so scale is an ops choice.

### R3 — EIG estimation is expensive and biased
- **Threat:** the planner's central quantity is a nested expectation; naive
  estimation is costly and can mislead experiment choice.
- **Mitigation:** the estimator ladder ([05 §4](./05-information-theory.md#4-estimators-eig-is-hard-be-honest-about-it))
  — closed-form for GP/linear-Gaussian, variational lower bounds (safe under
  maximization), surrogate acceleration; **every EIG carries its own standard
  error**, and the planner treats a noisy EIG as noisy.
- **Residual:** for high-dimensional non-Gaussian problems EIG stays hard; the
  planner degrades to honest wide error bars, not false confidence.

### R4 — Garbage-in modeling (the deepest limit)
- **Threat:** SDE cannot know a domain's likelihood or prior is *correct*; wrong
  models yield confident-but-wrong posteriors.
- **Mitigation:** make every modeling choice an explicit, versioned, criticizable
  object; support posterior-predictive checks and model criticism as stages;
  surface contradictions ([08 §5](./08-scirust-integration.md#5-contradiction-detection-concretely)).
- **Residual:** SDE guarantees *reasoning is consistent, reproducible, and
  auditable* — not that the science is right. This is stated as a
  [non-goal](./README.md#non-goals-for-v1), not papered over.

### R5 — Adoption / cold-start (the "Git of X" needs a community)
- **Threat:** infrastructure with no users is a museum piece.
- **Mitigation:** thin onboarding (one `Domain` trait), strong defaults via
  SciRust, MCP interop so existing Python/agent stacks plug in without a rewrite,
  and a compelling first artifact — SciRust's own reproduced research.
- **Residual:** network effects are earned, not designed; M4's "merge two labs'
  studies" is the feature that, if it lands, compounds.

### R6 — Untrusted plugin / experiment code execution
- **Threat:** third-party stages or executors are a security surface.
- **Mitigation:** capability-gated plugins ([07 §4](./07-extension-api-and-plugins.md#4-the-registry--capability-descriptors)),
  WASM sandbox with no ambient authority for untrusted algorithms, signed
  time-boxed `Capability` for effectful executors (the
  `scirust-discovery::ScopeAuthorization` pattern), every action pre-logged to a
  hash chain.
- **Residual:** out-of-process MCP backends run with the trust the operator
  grants them; SDE records *what* they did, and constrains *whether* they may.

### R7 — Non-Rust / FFI determinism leakage
- **Threat:** a Python or C backend introduces hidden nondeterminism (thread
  races, unseeded RNG, wall-clock).
- **Mitigation:** such backends declare L0/L1 honestly; record/replay pins their
  *outputs* even when their internals are opaque, so downstream stays
  reproducible against the recording.
- **Residual:** you cannot *re-derive* an L0 backend's output, only replay it —
  which is exactly the L0 contract, and disclosed.

### R8 — Scope creep / boiling the ocean
- **Threat:** "orchestrate all of science" is unbounded.
- **Mitigation:** vertical-slice roadmap; a frozen small ABI; every widening is a
  plugin, not a core change; explicit non-goals.
- **Residual:** governance discipline (the SDE-RFC process) is a human process
  and must be held.

### R9 — Correctness of the reasoning kernel itself
- **Threat:** a bug in hashing, memoization, or Bayes updating silently corrupts
  conclusions — worse than no tool.
- **Mitigation:** the kernel is small and pure; property-test the hash/canonical
  invariants and the memoization equivalence (`cache hit ⟺ recompute`); the
  `CpuBackend` oracle cross-checks numerics; a long-term goal is formal
  verification of the hashing + provenance core ([Part C](#part-c--future-research-directions)).
- **Residual:** trust in SDE is trust in a small verified core plus declared
  plugin levels — deliberately a *small* trusted computing base.

---

## Part C · Future research directions

Where SDE stops being engineering and becomes a research platform. Each ties to
capabilities already seeded in this workspace.

### F1 — Automated theory formation
Move from *fitting* hypotheses to *generating their structure*: symbolic
regression (`scirust-symreg`) and program synthesis (`scirust-synthesis`,
`scirust-neuro-symbolic`) as `HypothesisGenerator`s that propose *mechanisms*,
not just parameters — with the DAG keeping every machine-proposed law
falsifiable and provenance-bound.

### F2 — LLM-in-the-loop, kept honest by the graph
`scirust-sciagent` (or any LLM) as a hypothesis generator and question
decomposer over MCP — but every suggestion enters as an ordinary, criticizable
`Hypothesis` with its prompt and seed recorded. The open question: **how much
does auditable, provenance-bound LLM ideation improve discovery rate without
importing the reproducibility crisis it usually causes?** SDE is the harness that
can measure this.

### F3 — Non-myopic sequential experiment design
The default planner is one-step-greedy. Finite-horizon BOED / active learning
that plans *sequences* of experiments (lookahead over the belief tree) trades
compute for better long-run information yield — a natural `SequentialPlanner`
plugin and a rich research direction ([05 §6](./05-information-theory.md#6-the-planner-interface)).

### F4 — Causal discovery
Extend hypotheses from associational models to **causal** ones; let experiment
design exploit interventional identifiability (choose the intervention that most
disambiguates causal structure). EIG over causal graphs is a direct fit for the
existing planner.

### F5 — Federated & collaborative discovery ("pull requests for science")
Content-addressing already makes two labs' graphs *mergeable*. The research
question is the **social protocol**: how do independent groups reconcile
conflicting `Confidence` objects, weight each other's L0 observations, and build
a shared, signed, federated discovery ledger? This is the feature that could make
SDE infrastructure rather than a tool.

### F6 — Formal verification of the reasoning kernel
Prove the core invariants — canonical-serialization determinism, hash
collision-resistance assumptions, memoization equivalence, monotone determinism-
level propagation — so the trusted computing base is *verified*, not merely
tested. Aligns with the workspace's `scirust-func-safety` certification culture.

### F7 — Cross-domain prior transfer
When two domains share structure (an oscillator in physics and in a control
system), can a posterior learned in one seed a prior in the other? Content-
addressed, typed models make principled transfer *findable*; whether it helps is
empirical — and SDE can run that meta-experiment on itself.

### F8 — Self-application (SDE studies SciRust)
The sharpest dogfooding: point SDE at SciRust's *own* open research questions
(the `docs/research/` programs — SRCC, ANEE, quantum) and let the planner
recommend which benchmark or ablation buys the most information next. A
scientific-discovery engine whose first non-trivial subject is its own backend's
development is both a proof of generality and a genuine research accelerator —
the `scirust-rsi` (recursive self-improvement) direction, made reproducible and
auditable.

---

## Closing

SDE's bet is narrow and large at once: that the scientific method is a
computation, and that treating it as one — with immutable content-addressed
objects, an explicit effect boundary, a determinism taxonomy that never lies, and
an information-theoretic planner — buys reproducibility, composability, and
automation the way Git and LLVM did for their domains. The object model is the
commitment; everything else is a plugin. If the walking skeleton of M1 reproduces
this repository's own research history, the thesis has its first data point.

---

> [← Workspace & Crates](./09-workspace-and-crates.md) · [Back to README](./README.md)
