# Causal and Experimental Structural Intelligence Program

Tracking document for the program that extends SciRust's structural-intelligence
line — measure → compare → abstain/select → calibrate → certify → deploy →
monitor → rollback, closed by the
[SRCC Robust Structural Intelligence Program](SRCC_ROBUST_STRUCTURAL_INTELLIGENCE_PROGRAM.md)
(Program 4) — into a tenth stage: observe → distinguish association from causal
evidence → represent assumptions → discover equivalence classes → estimate
identifiable effects → test invariance → simulate interventions → choose the
next experiment → update theories → verify causal claims.

This file is maintained incrementally, in the same spirit as its predecessor:
each phase appends its design summary, merge commit, and known limitations. It
does **not** modify Program 4's closing synthesis or conclusion — that program
is finished; this is a new one that builds on top of it.

## The mandate this program exists to enforce

**Predictive or optimization success must never be converted into an
unjustified causal claim.** Every capability this program adds must be able to
say, honestly, one of:

- "Under assumptions `A`, using evidence `E`, property `Q` is **identifiable**,
  estimated by `M`, with uncertainty `U`, sensitivity `S`, and unresolved
  alternatives `R`"; or
- "Under assumptions `A`, using evidence `E`, property `Q` is
  **not identifiable** / only an **equivalence class** / **inconclusive**" —
  and stop there.

A fitted model that predicts well, or a discovery algorithm that converges, is
not thereby a causal oracle. See [`scirust-causal`'s crate-root
documentation](../../scirust-causal/src/lib.rs) ("Causal interpretation — read
before using the discovery API") for the same rule stated at the code level.

## Program-wide invariants

These hold in every phase and are re-checked in each PR's self-review — the
same discipline Program 4 used, unchanged:

- **Pure Rust.** No FFI, no network access in library code or tests,
  `#![forbid(unsafe_code)]` in every crate this program touches.
- **Deterministic.** Fixed accumulation order, explicit recorded seeds,
  canonical sorting by `f64::total_cmp` with stable keys where sorting is
  needed, no thread-scheduling-dependent reductions.
- **Typed errors.** Error enums use manual `Display` + `Error` impls (the
  established SciRust convention), never a stringly-typed catch-all.
- **Backward-compatible by default.** Existing public APIs stay
  source-compatible; new behavior is opt-in. Prefer extending an existing crate
  over creating a new one.
- **MSRV 1.89.**
- **Leakage-free.** Any evaluation protocol this program adds follows the
  no-leakage discipline Program 4 established (727–728).
- **Certificate-driven.** A causal conclusion is a typed
  [`CausalCertificate`](../../scirust-causal/src/certificate.rs), never a bare
  number or a prose claim.
- **Safe to abstain.** Non-identifiability, non-convergence, and "only an
  equivalence class" are first-class results, reachable and tested — never
  swallowed into a false positive.

## Honesty rules

1. **Association is not causation.** A statistical dependency (correlation,
   predictive skill, low training loss) is never itself reported as a causal
   effect.
2. **Discovery returns equivalence classes, not single DAGs**, whenever the
   evidence only supports that. A CPDAG/PAG representative is exactly that — a
   representative — never presented as *the* causal graph.
3. **Effect identification requires stated assumptions.** No identifiability
   claim is made without naming, in a
   [`CausalAssumption`](../../scirust-causal/src/assumptions.rs) registry, what
   is being assumed and why ([`AssumptionBasis`](../../scirust-causal/src/assumptions.rs)).
4. **Counterfactuals require a structural causal model (SCM).** Simulating an
   intervention's downstream effect is only ever claimed relative to a stated
   SCM, never inferred from observational fit alone.
5. **Negative outcomes are first-class.** `NotIdentifiable`, `Inconclusive`,
   `EquivalenceClassOnly` (see
   [`IdentifiabilityStatus`](../../scirust-causal/src/certificate.rs)) are
   ordinary, tested, expected results — not failure modes to be designed away.

## Protocol

- Each phase is a separate PR, branched from a **newly-merged** `master`. A
  later phase never starts from an unmerged earlier one.
- No PR auto-merges without explicit authorization and green CI.
- If a capability already exists, `master` has advanced past what a phase
  assumed, a result is provably not identifiable, only an equivalence class is
  available, or a planned algorithm (e.g. FCI under latent confounding) would
  be incomplete for the stated assumptions — **report it and adjust scope
  rather than silently weakening the design** to force a phase to "succeed."

## Foundation: `scirust-causal` (pre-existing, audited and remediated)

Before this program's own phases began, `scirust-causal` already existed as a
separate contribution (PR #807): deterministic invertible cubic flows
(`TriangularCubicFlow`) and a NOTEARS-style continuous causal-structure
optimizer (`optimize_causal`, `CubicCausalScore`, `PolynomialAcyclicity`,
`extract_causal_dag`). This program's repository audit found it, and per an
explicit decision to build on rather than duplicate it, performed an
adversarial review before adopting it as the substrate for phase 5C.1.

**What the review found and fixed** (PR #810, merged as commit `a16e9a43` —
note PR #807 itself merged the pre-remediation code first via an unrelated
automated process, so #810 landed the fix directly against `master` rather
than against #807, which had already closed):

- The all-zeros interaction matrix is a **saddle point** of the cubic score
  (its gradient vanishes there identically), so a zero-initialized optimizer
  took zero descent steps and the empty graph was silently reported as
  `Converged`. Fixed by adding `TerminationReason::StationaryAtInitialPoint` as
  a distinct, first-class, tested outcome.
- Non-convergence was swallowed into `Ok(..)` with no signal to the caller.
  Fixed: every non-convergence path now returns a specific
  `TerminationReason` plus a `warnings` log, never a bare success.
- The crate's documentation asserted capabilities and convergence semantics
  the implementation didn't actually provide. Rewritten around a "Causal
  interpretation — read before using the discovery API" section stating the
  identifiability caveats this whole program exists to formalize.
- Untested triangularization, a `0 * inf = NaN` slip-through in the
  acyclicity gradient's finiteness guard, and dead error variants were fixed;
  the end-to-end test was rewritten to demonstrate — as an executable check,
  not just prose — that this discovery method is **non-identifiable**: two
  initializations on the same data converge to two different feasible DAGs,
  neither the true generating chain.

This is the honest state phase 5C.1 builds on: a sound, deterministic
optimizer that finds *a* feasible graph, with no claim that it finds *the*
graph.

## Phase roadmap

The ten conceptual stages this program targets, mapped to phases. **Only
5C.1's scope below is final** — later phases are a provisional roadmap, scoped
in full detail only when they are actually started (per the protocol above,
each begins from newly-merged `master`, so a later phase's exact shape may
shift with what's true at the time).

| Phase | Conceptual stage | Status |
| --- | --- | --- |
| — | Observe | Pre-existing (`scirust-causal`, audited & remediated above) |
| 5C.1 | Represent assumptions | **Done** — typed causal contracts and data model |
| 5C.2 | Distinguish association from causal evidence | Planned — conditional-independence testing |
| 5C.3 | Discover equivalence classes | Planned — CPDAG/PAG-returning discovery |
| 5C.4 | Estimate identifiable effects | Planned — adjustment-set estimation under stated assumptions |
| 5C.5 | Test invariance | Planned — cross-environment invariant prediction |
| 5C.6 | Simulate interventions | Planned — SCM-based counterfactual simulation |
| 5C.7 | Choose the next experiment | Planned — experimental design / value of information |
| 5C.8 | Update theories | Planned — assumption-registry revision under new evidence |
| 5C.9 | Verify causal claims | Planned — end-to-end certificate audit |
| 5C.10 | Closing synthesis | Planned |

## Phase 5C.1 — Typed causal contracts and data model

**Status: done.** Additive to `scirust-causal` (no existing public API
changed). No discovery, identification, or estimation algorithm is introduced
in this phase — it defines contracts, not procedures.

### Design

Nine new modules, all reusing `scirust-graph::dag::CausalDag` and
`scirust-solvers::Matrix` rather than duplicating graph or linear-algebra
substrate:

- **`variable.rs`** — `CausalVariable` (positional `index`, `name`, `role`,
  `kind`), `VariableRole` (Treatment/Outcome/Covariate/Confounder/Mediator/
  Instrument/Collider/Unspecified — relative to a query, not intrinsic to the
  variable), `VariableKind` (Continuous/Discrete/Binary), and
  `validate_variable_set` (indices are exactly `0..n` with no gaps/duplicates,
  matching `CausalDag` node-id and `CausalDataset` column conventions).
- **`intervention.rs`** — `InterventionKind` (`Atomic` = Pearl's `do(X=x)`,
  `Shift` = additive mechanism-preserving shift, `MechanismChange` = a known
  but unparameterized regime change, `Unspecified`), `Intervention` (target +
  kind, validated finite).
- **`environment.rs`** — `Environment`: a labeled data-generating regime
  (an id plus zero or more simultaneous interventions on distinct targets),
  the precondition later invariance-testing phases (5C.5) operate on.
- **`dataset.rs`** — `SampleBlock` (row-major samples tagged with an
  `Environment`; converts to/from `Matrix`, since `Matrix` itself has no
  `serde` support) and `CausalDataset` (a variable set plus one or more
  blocks, plus a free-text provenance `source` string). Validates block/variable
  dimension agreement and that every intervention target is in range.
- **`assumptions.rs`** — `CausalAssumption` (a closed, named set —
  Acyclicity, CausalSufficiency, Faithfulness, CorrectFunctionalForm,
  AdequateSampleSize, Sutva, Exchangeability, Positivity,
  InvarianceAcrossEnvironments — plus an `Other(String)` escape hatch),
  `AssumptionBasis` (the **provenance**: AssertedByAnalyst,
  GuaranteedByDesign, TestedStatistically, DomainKnowledge, or the safe
  default `Unverified`), and `AssumptionRegistry` — `BTreeMap`-keyed so
  iteration order is deterministic regardless of insertion order (this is
  what makes a certificate built from a registry fingerprint-stable).
  Re-asserting a registered assumption is a validation error; replacing one
  requires the explicitly-named `overwrite`.
- **`graph_constraints.rs`** — `GraphConstraints`: required/forbidden edges and
  a partial tier (temporal) order over `n_variables`, with mutual-consistency
  checks at insertion time (can't require and forbid the same edge; a tier
  assignment that would retroactively violate an existing required edge is
  rejected and rolled back). `GraphConstraints::check` validates a candidate
  `CausalDag` against this background knowledge and is panic-safe against a
  DAG smaller than `n_variables`.
- **`certificate.rs`** — `IdentifiabilityStatus` (Identifiable,
  NotIdentifiable, EquivalenceClassOnly, Inconclusive — every variant a
  legitimate, equally-weighted outcome) and `CausalCertificate` /
  `CausalCertificateBuilder`. The builder is the **only** construction path,
  and `finalize()` enforces the one rule this type exists to make impossible
  to violate: **only `Identifiable` may carry a numeric estimate.** Attempting
  to attach an estimate to any other status is a construction error, not a
  value the type will silently hold. `assumptions_used` and
  `unresolved_alternatives` are sorted and deduplicated before finalizing, so
  the certificate's identity (and fingerprint) does not depend on the order
  the caller happened to list them in.
- **`fingerprint.rs`** — `sha256_hex`, mirroring the convention already
  established in `scirust-srcc-bench::records::sha256_hex`. A certificate's
  fingerprint commits to every semantic field except itself (a private
  `CertificatePreImage` excludes the fingerprint field to avoid
  self-reference).

### Determinism contract

- `AssumptionRegistry` iterates in `CausalAssumption`'s `Ord` order — a
  `BTreeMap`, never a `HashMap` — so two registries built by asserting the
  same entries in different orders iterate identically (tested).
- `CausalCertificateBuilder::finalize` sorts and dedupes both
  order-insensitive fields before hashing, so its fingerprint is order
  invariant over `assumptions_used` (tested) and is bit-identical across
  repeated builds of the same content (tested) and across process runs (the
  `typed_causal_contract` example prints its fingerprint; running it twice
  produces the same digest, verified during validation).
- No `Date.now`/random seed/timestamp participates in any typed-contract
  type; the only randomness anywhere in the crate remains the pre-existing,
  explicitly-seeded `SplitMix64` synthetic-data generator.

### Tests

97 tests existed before this phase (`scirust-causal`, post-#810); this phase
adds 76 across seven new test files (`variable.rs`, `intervention.rs`,
`environment.rs`, `dataset.rs`, `assumptions.rs`, `graph_constraints.rs`,
`certificate.rs`), covering: construction validation (every documented error
path), JSON round-trip (byte-exact on embedded `f64` sample data, via
`serde_json`'s `float_roundtrip` feature), the coherence rule (all four
non-`Identifiable` statuses independently confirmed to reject an attached
estimate; `Identifiable` confirmed to accept one), fingerprint determinism
and order-independence, `GraphConstraints`'s mutual-consistency and
panic-safety-against-a-smaller-DAG, and integration with the pre-existing
synthetic-data pipeline (`wraps_existing_synthetic_pipeline_output`).

The `examples/typed_causal_contract.rs` example runs the existing (unmodified)
`optimize_causal` → `extract_causal_dag` pipeline on a known synthetic chain,
wraps every stage in the new typed contracts, and reports the result as
`EquivalenceClassOnly` — not `Identifiable` — because nothing in this phase
performs the identifiability reasoning that would justify a stronger claim. It
also demonstrates the coherence rule firing on a deliberately-wrong attempt to
attach an estimate to that status.

### Compatibility

Purely additive: nine new modules, no existing public item changed, three new
dependencies (`serde`, `serde_json` with `float_roundtrip`, `sha2` — all
already used elsewhere in the workspace at the same version bounds, e.g.
`scirust-srcc-bench`).

### Known limitations / deferred

- No conditional-independence testing, discovery algorithm, effect
  estimation, SCM, or invariance test exists yet — those are 5C.2 onward.
  Nothing in this phase can actually populate an `Identifiable` certificate
  with a real estimate; the type only guarantees that when something *does*,
  it cannot do so incoherently.
- `GraphConstraints` background knowledge (required/forbidden edges, tiers)
  is not yet consumed by any discovery procedure — `extract_causal_dag`
  (pre-existing) does not take a `GraphConstraints` argument. Wiring that in
  is for whichever later phase adds a constrained discovery algorithm.
- `AssumptionBasis::TestedStatistically` records that a test was run and its
  name/p-value; it does not run any test itself. That is 5C.2.
- The closed `CausalAssumption` variant set reflects the assumptions named in
  `scirust-causal`'s own crate-root documentation; the `Other(String)` escape
  hatch exists precisely because later phases will likely need assumptions
  not yet named here (e.g. positivity-of-instrument strength, monotonicity
  for IV estimators).
