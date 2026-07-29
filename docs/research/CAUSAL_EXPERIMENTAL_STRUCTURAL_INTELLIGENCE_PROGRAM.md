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
| 5C.2 | Distinguish association from causal evidence | **Done** — deterministic robust conditional-independence testing |
| 5C.3 | Discover equivalence classes | **Done** — PC-Stable, CPDAG-returning (no PAG/latent-confounding-robust discovery) |
| 5C.4 | Estimate identifiable effects | **Done** — backdoor identification + linear adjustment estimation |
| 5C.5 | Quantify sensitivity to unmeasured confounding | **Draft** — Cinelli–Hazlett omitted-variable-bias bounds |
| 5C.6 | Test invariance | Planned — cross-environment invariant prediction |
| 5C.7 | Simulate interventions | Planned — SCM-based counterfactual simulation |
| 5C.8 | Choose the next experiment | Planned — experimental design / value of information |
| 5C.9 | Update theories | Planned — assumption-registry revision under new evidence |
| 5C.10 | Verify causal claims | Planned — end-to-end certificate audit |
| 5C.11 | Closing synthesis | Planned |

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

## Phase 5C.2 — Deterministic robust conditional-independence testing

**Status: Done.** Branch `claude/scirust-srcc-robust-stats-6ue9xc`, opened
from `origin/master` at `5fd76dcc` (the commit 5C.1 merged onto, unchanged by
this phase). PR #821, merged at `1fd2ffef`. Additive to `scirust-causal` (no
existing public API changed). This phase implements **statistical testing
only** — it does **not** implement PC-Stable or any other causal
graph-discovery algorithm, does not compute a CPDAG/PAG equivalence class,
and does not estimate a causal effect. It builds the statistical oracle
(`X ⟂ Y | Z` evidence) a discovery algorithm would later consume — see 5C.3
in the roadmap above for that (still unstarted) next step.

### Scientific scope — read before using this API

A [`scirust_causal::PartialCorrelationTest`] answers one narrow statistical
question: is the *linear* partial association between `X` and `Y`,
controlling for `Z`, distinguishable from noise under a stated model,
calibration, and significance level? It does **not** establish that a causal
edge is absent, that the eventual discovered graph is acyclic, causal
sufficiency, faithfulness, the absence of selection bias, or correct temporal
ordering — see the crate root's "Causal interpretation" section and the
(private) `conditional_independence` module's own docs, which this phase's
results are subject to exactly the same way. Three further, deliberately
undisguised limitations, each with its own adversarial test:

- **Linear only.** A linear partial correlation can be exactly zero while `X`
  and `Y` remain conditionally *dependent* through a nonlinear or purely
  heteroscedastic relationship (`nonlinear_dependence_is_invisible_to_a_linear_partial_correlation_test`,
  `heteroscedastic_dependence_is_invisible_to_a_mean_based_linear_test`).
- **Failure to reject is not proof of independence.**
  `IndependenceDecision::IndependentWithinThreshold` means exactly that — the
  null was not rejected under the declared model/calibration/alpha/sample —
  never that independence was established. A tiny but real path coefficient
  can be statistically invisible at ordinary sample sizes
  (`near_unfaithful_chain_with_tiny_coefficient_is_not_reliably_detected`).
- **Latent confounding is untestable by construction.** If a confounder is
  not a column in the dataset, no conditioning-set choice can control for it;
  the test cannot and does not distinguish a direct causal link from a
  latent-confounded one
  (`confounded_association_with_latent_confounder_cannot_be_told_apart_from_direct_dependence`).

### Design

Four new modules, reusing existing infrastructure rather than duplicating
it — QR/SVD from `scirust-solvers`, the standard-normal survival function
from `scirust-stats`, and the OGK robust scatter estimator from
`scirust-multivariate` (Program 4, phase 4E.1) are all consumed, not
reimplemented:

- **`partial_correlation.rs`** (private) — fixed-order-accumulation Pearson
  correlation; QR-residualization (`scirust_solvers::linalg::qr_decompose` /
  `solve_qr_least_squares`) against an intercept + conditioning-set design,
  with numerical rank checked via SVD *before* solving (typed
  `RankDeficientConditioningSet` on a design below full column rank, not a
  silent least-squares fallback); Fisher-z calibration
  (`z = atanh(r)·sqrt(n − |Z| − 3)`, `None` — not an error — when degrees of
  freedom are exhausted or `|r| = 1` exactly).
- **`robust_partial_correlation.rs`** (private) — the robust analogue, in two
  stages: (1) OGK-projection residualization against `Z` (reusing
  `RobustScatterModel::inverse_scatter`'s conditional-mean identity, exactly
  as documented in the crate); (2) a **second**, two-dimensional OGK fit on
  the two residual vectors, reading the correlation directly off *that* fit's
  precision matrix (`r = -P[0,1] / sqrt(P[0,0]·P[1,1])`). Stage 2 matters: an
  earlier iteration of this phase computed the final correlation via ordinary
  Pearson correlation of the (robustly centered) residuals, which is
  mathematically inert — Pearson recenters internally, so any prior centering
  cannot change its output — silently providing **zero** actual robustness
  for the empty-conditioning-set case. This was caught by a comparative test
  (contaminated data giving identical classical/robust statistics) before
  being shipped, and is now pinned down by a permanent regression test
  (`contaminated_empty_z_result_differs_from_plain_pearson_correlation`). On
  genuinely clean data this method is routinely *bit-identical* to the
  classical one — an expected, correct property of `RobustScatterConfig`'s
  default hard-reweighting OGK (when no row is rejected, the reweighted
  scatter *is* the ordinary covariance of every row), not a bug or an unused
  code path; the two visibly diverge once rows are actually down-weighted.
  `RobustCalibration::NoPValue` (the honest default) reports the statistic
  with no p-value; `GaussianApproximation` applies the same Fisher-z formula
  with an always-attached inexactness warning (not proven exact for an
  OGK-derived statistic); `Permutation` calibrates deterministically.
- **`permutation_calibration.rs`** (private) — one continuing
  `scirust_stats::SplitMix64` stream drives all `B` requested permutations
  (Durstenfeld Fisher-Yates on `0..n`); each permutation reshuffles a
  **residual** (Freedman-Lane-style), not a raw variable — naive raw-variable
  permutation is invalid whenever the permuted variable actually depends on
  `Z`, so this module never offers that as an option. Two-sided p-value
  `(1 + exceedances) / (1 + completed)` — the standard finite-sample
  correction, never exactly zero. A permutation whose recomputation
  degenerates (e.g. a zero-variance resample, or — for the robust path — a
  singular 2-D refit) is excluded from both `completed` and `exceedances`,
  never silently treated as a non-exceedance.
- **`conditional_independence.rs`** (private, re-exports public) — the
  orchestration layer: `ConditionalIndependenceTest` trait,
  `PartialCorrelationTest` (the one implementor this phase ships),
  `ConditionalIndependenceConfig` (validated `significance_level ∈ (0,1)`,
  rank tolerance, `RegimeSelection`, `MissingValuePolicy`),
  `ConditionalIndependenceMethod` (Gaussian / Robust / Permutation, each
  carrying its own calibration choice), and `ConditionalIndependenceResult`
  (`x`, `y`, canonicalized `conditioned_on`, `statistic`, `effect_size`,
  `p_value: Option<f64>`, `decision`, `sample_count`, `effective_rank`,
  `method`, `calibration`, `assumptions`, `warnings`). `IndependenceDecision`
  is a **three-way** outcome (`Dependent` / `IndependentWithinThreshold` /
  `Inconclusive`), never collapsed to a boolean, and is kept structurally
  distinct from a typed `CausalError` (malformed *inputs* — unknown/duplicate/
  endpoint-overlapping variable, insufficient samples, non-`Continuous` kind —
  are errors; a well-formed but scientifically unresolved request is
  `Inconclusive`, never an error). `RegimeSelection` (ObservationalOnly /
  Environment(id) / ExplicitRows) makes mixing interventional and
  observational rows an explicit, auditable choice rather than a silent
  default; `MissingValuePolicy` (Error / CompleteCases) is implemented and
  tested even though `CausalDataset`'s current finite-at-construction
  invariant makes it presently a no-op — the no-op-ness is itself a checked
  claim, not an assumption.
- **9 new `CausalError` variants** (extending the crate's existing one error
  enum, per its established one-enum-per-crate convention, rather than
  introducing a parallel `ConditionalIndependenceError`): `SameVariable`,
  `ConditioningContainsEndpoint`, `DuplicateConditioningVariable`,
  `UnsupportedVariableKind`, `InsufficientSamples`, `NonFiniteSample`,
  `ZeroVariance`, `RankDeficientConditioningSet`, `ScatterFailure` (wraps
  `scirust_multivariate::RobustGeometryError` as a real `source()`, not a
  stringified message), `SolverFailure`. One new `CausalAssumption` variant,
  `ResidualExchangeability` — the precondition Freedman-Lane permutation
  relies on.
- Two new dependencies in `scirust-causal/Cargo.toml`: `scirust-stats` and
  `scirust-multivariate` (both path dependencies, already at the top of the
  dependency graph — `scirust-multivariate` depends on `scirust-stats`, which
  depends only on `scirust-special`; neither depends back on
  `scirust-causal`, so no cycle is introduced).

### Determinism contract

- The conditioning set is canonicalized (sorted) before any computation, so
  callers passing the same set in a different order get identical results
  (tested for all three methods, including the permutation-calibrated one).
- Row selection and column extraction use a fixed block-then-row order; QR/SVD
  and OGK are both deterministic by construction (no internal RNG, fixed
  accumulation order).
- The one seeded procedure (permutation calibration) is a single continuing
  `SplitMix64` stream, entirely determined by `seed` and the sample count.
- No floating-point sort occurs anywhere in this phase's code: SVD already
  returns singular values pre-sorted descending, and exceedance counting is a
  direct `>=` comparison on already-validated-finite values — so
  `f64::total_cmp` is not needed here.
- `examples/conditional_independence_benchmark.rs` is deterministic
  end-to-end (fixed seeds, no wall-clock/hostname in its stdout). Run twice
  and hashed:

  ```
  SHA-256 (scientific stdout, nightly-2026-07-02, x86_64):
  c1449177f21aad6c7579bf5de902e654531c8e3d0c195ae88a4530d6b0ab7a9c
  ```

  (Confirmed bit-identical across two consecutive runs, and across a debug
  vs. release build.) The historical `industrial_protocol_demo` fingerprint
  (`167c13de…`) was independently reverified unchanged — this phase touches
  no file that example depends on.

### Tests

166 tests existed for `scirust-causal` before this phase (verified directly
against `origin/master` at `5fd76dcc`, not assumed from prior phases' notes);
this phase adds **82**: 26 embedded unit tests across the three new private
modules, 29 in `tests/conditional_independence.rs` (basic correlation cases,
the three causal motifs — chain/fork/collider, each with the theoretically-
predicted marginal/conditional (in)dependence pattern verified, including the
collider's "conditioning induces dependence" case future discovery algorithms
rely on — confounded association with an observed vs. latent confounder,
9 dataset-contract checks, JSON round-trip, and 6 property-style invariance
tests: symmetry, conditioning-set-order, row-order, positive-scale,
translation, and sign-negation invariance), and 27 in
`tests/conditional_independence_adversarial.rs` (contamination: vertical
outliers, bad leverage points, correlated/structured contamination, a clean
case where classical and robust agree, bitwise-deterministic robust repeats,
a near-constant conditioning dimension; permutation calibration: determinism,
seed-sensitivity of the p-value without a change in result shape, the exact
two-sided exceedance formula, detection of real dependence, non-rejection of
real independence, the chain motif via residual permutation, an invalid
permutation count, conditioning-order invariance; boundary/numerical cases:
near-perfect vs. exact rank deficiency, a conditioning set that saturates the
sample — proven to force a spurious `r = ±1` that is honestly reported
`Inconclusive`, not `Dependent` — a near-unfaithful (tiny-coefficient) chain,
a minority bypass-contaminated conditional test, heavy-tailed independent
variables, the two undisguised nonlinear/heteroscedastic negative results,
mixed intervention/observational rows, a small environment at the exact
sample-size boundary, and duplicate variable metadata).

### Compatibility

Purely additive: four new (private) modules plus new public re-exports
(`PartialCorrelationTest`, `ConditionalIndependenceTest`,
`ConditionalIndependenceConfig`, `ConditionalIndependenceMethod`,
`ConditionalIndependenceResult`, `IndependenceDecision`, `CalibrationMethod`,
`RegimeSelection`, `MissingValuePolicy`, `ResidualizationMethod`,
`RobustCalibration`), 9 new `CausalError` variants and 1 new
`CausalAssumption` variant (both additive to existing open enums, not
breaking), 2 new path dependencies. No existing public item's signature
changed; `examples/typed_causal_contract.rs` is untouched and its behavior is
unaffected.

### Supported and unsupported claims

May claim: deterministic linear conditional-independence testing under
(approximate) Gaussian assumptions with Fisher-z calibration; a genuinely
robust association measure via OGK when data is contaminated; a calibrated
p-value under a documented, named exchangeability assumption via residual
permutation; a structurally-enforced three-way decision (never a boolean);
suitability as one candidate statistical input to a future PC-Stable-style
discovery algorithm.

Must **not** claim: that `IndependentWithinThreshold` proves independence or
that a graph edge is absent; that latent confounding has been excluded;
that faithfulness has been validated; that the classical Fisher-z null is
exact for an OGK-derived statistic; that permutation calibration is valid
under every possible dependence structure (only under
`ResidualExchangeability`); that this phase detects arbitrary nonlinear
dependence; that a DAG has been discovered or that any effect is
identifiable or estimated.

### Known limitations / deferred

- Linear association only — see "Scientific scope" above; a future phase
  that wants nonlinear CI testing (e.g. kernel-based or rank-based measures)
  would need a new method variant, not a change to this one.
- The robust method's residualization uses two independent per-variable OGK
  fits against `Z`, then a third 2-D fit on the residuals — not a single
  joint fit over `{X, Y} ∪ Z` with the partial correlation read off in one
  step. Both are legitimate designs; this phase does not claim the two are
  numerically equivalent.
- `MissingValuePolicy` is a no-op under `CausalDataset`'s current
  finite-at-construction invariant (inherited from 5C.1, unchanged here).
- No conditional-independence-based discovery algorithm (PC-Stable or
  otherwise), equivalence-class construction, effect estimation, or
  invariance test exists yet — those remain 5C.3 onward, not to be started
  until this phase is merged and `master` is resynchronized.

## Phase 5C.3 — Discover equivalence classes (PC-Stable)

**Status: Done.** Branch `claude/scirust-srcc-robust-stats-6ue9xc`, restarted
from `origin/master` at `376fd353` (fresh master after PR #821 merged; this
phase's branch carries only this phase's commits). PR #824, merged at
`a7214617`. Additive to
`scirust-causal` (no existing public API changed). This phase implements
**constraint-based Markov-equivalence-class discovery** — PC-Stable (Colombo &
Maathuis, *Order-Independent Constraint-Based Causal Structure Learning*,
JMLR 2014), the order-independent variant of Spirtes, Glymour & Scheines's PC
algorithm — built entirely on top of 5C.2's conditional-independence oracle.
It does **not** implement FCI or any latent-confounding-robust discovery, does
**not** construct a PAG, and does **not** estimate a causal effect.

### Scientific scope — read before using this API

`PcStable::discover` answers: *given repeated conditional-independence
evidence, what is the Markov equivalence class consistent with it?* Under
three assumptions — **acyclicity**, **causal sufficiency** (no latent
confounder between any two observed variables), and **faithfulness** (every
conditional independence in the data reflects a d-separation in the true
graph, no coincidental cancellation) — and given a CI oracle without error,
this procedure recovers the *exact* equivalence class: every directed edge in
the output [`Cpdag`](../../scirust-causal/src/cpdag.rs) is compelled in every
DAG consistent with the observed (in)dependencies; every undirected edge is
genuinely ambiguous from this evidence alone.

It must **not** be read as: proof that causal sufficiency holds (see the
latent-confounding adversarial test below — a confounded pair looks *exactly*
like an ambiguous-direction direct edge, and this procedure has no way to
tell the difference); proof that faithfulness holds; a claim that an
undirected edge means "no causal relationship" (it means the opposite — a
causal relationship whose direction the data cannot determine); or immunity
from the standard bounded-conditioning-set-size limitation any constraint-
based search shares (a true separating set larger than
`EquivalenceClassConfig::max_conditioning_set_size` is missed, incorrectly
retaining that edge).

### Design

Three-stage pipeline, one file per stage plus the public orchestration layer:

- `cpdag.rs` — [`Cpdag`]: a plain, invariant-protected partially directed
  graph (`BTreeSet<(usize,usize)>` for directed edges, a second one,
  canonicalized `(min,max)`, for undirected edges — a pair is never in both).
  Fields are private; mutation (`orient`, `remove_edge`) goes only through
  methods that preserve the invariant.
- `skeleton_discovery.rs` — the adjacency search. Starting from the complete
  graph, for increasing conditioning-set size `ℓ = 0, 1, 2, …`: freeze every
  variable's adjacency into a snapshot at the *start* of the level (the
  "stable" fix — see the module docs for exactly why classic PC's live-updated
  adjacency is order-dependent and this snapshot removes that dependence);
  test every still-adjacent pair against size-`ℓ` subsets of each endpoint's
  frozen neighbor set; remove every pair a test found
  `IndependentWithinThreshold` for, but only *after* the whole level
  finishes. `Dependent` and `Inconclusive` both leave an edge in place — only
  an explicit `IndependentWithinThreshold` verdict removes one. A handful of
  expected-at-large-`ℓ` [`CausalError`] variants (rank deficiency,
  insufficient samples, zero residual variance, a singular robust scatter, a
  solver failure) are caught, recorded as a warning, and treated as "this one
  candidate is untestable, try the next" — every other `CausalError` (a
  malformed request this module's own index bookkeeping should never
  produce) propagates as a genuine `Err`.
- `orientation.rs` — v-structure detection (every unshielded triple `x-z-y`
  whose recorded separating set for `{x,y}` does not contain `z` is a
  collider: orient `x->z`, `y->z`) followed by Meek's rules R1-R3 (Meek, UAI
  1995) applied to a fixpoint, propagating those orientations as far as they
  logically force without creating an unevidenced collider or a directed
  cycle. Two conflicting v-structure demands on the same edge (only possible
  under finite-sample error or a genuine assumption violation, provably not
  under a perfect oracle) are left undirected with a recorded warning, never
  silently resolved. Rule 4 is out of scope by construction: it is needed
  only when orientations *beyond* v-structures (background knowledge) are
  injected, and this phase accepts none.
- `equivalence_class.rs` — [`PcStable`] /
  [`EquivalenceClassDiscovery`] / [`EquivalenceClassConfig`] /
  [`EquivalenceClassResult`], wiring the three stages together and unioning
  the discovery procedure's own three assumptions with every underlying CI
  test call's own reported assumptions (so the final assumption list is
  honest about the *entire* evidentiary chain, not only the discovery
  procedure's own preconditions).

`EquivalenceClassResult::separating_sets` is a sorted `Vec<((usize,usize),
Vec<usize>)>`, not a `BTreeMap` — `serde_json` rejects a non-string map key at
serialize time (verified directly: `BTreeMap<(usize,usize), _>` fails with
"key must be a string"), so the public result type avoids that shape
entirely; the internal `BTreeMap` (used for O(log n) lookup during
v-structure detection) never crosses the public API.

This is a **separate, additive discovery paradigm** from the crate's existing
continuous-optimization structure learner (`optimize_causal`, constraint-based
vs. score-based); neither calls the other, and the crate root's docs are
updated to state precisely which capability now covers the
equivalence-class gap the optimizer's own docs have always named as out of
scope for itself.

### Determinism contract

Skeleton discovery's frozen-per-level adjacency makes the result independent
of the order pairs are visited *within* the algorithm (the "stable" property,
verified indirectly via a relabeling-invariance test — see below). All three
data structures (`Cpdag`'s `BTreeSet`s, `BTreeMap` separating sets) iterate in
a fixed, deterministic order; combinations of a frozen neighbor set are
generated in lexicographic order over an explicitly sorted slice. No RNG is
used anywhere in this phase's own code — determinism (or its absence) is
entirely inherited from whichever `ConditionalIndependenceTest` the caller
supplies (5C.2's own determinism contract already covers that).

### Tests

248 tests existed for `scirust-causal` before this phase (166 pre-5C.2 +
82 from 5C.2, verified against merged `master`). This phase adds **60**: 21
embedded unit tests (6 in `cpdag.rs`, 8 in `skeleton_discovery.rs` including
two full chain/collider end-to-end recoveries, 12 in `orientation.rs`
covering v-structure detection, a hand-verified conflicting-demand case, and
each of R1/R2/R3 in isolation — 3 in `equivalence_class.rs`), 7 in
`tests/pc_stable.rs` (chain/fork/collider motifs, the chain≡fork Markov-
equivalence demonstration, a 4-node case that hand-verifiably requires Meek's
rule 1 to complete, a 5-node two-chained-collider case resolved by
v-structures alone, cross-method compatibility with the robust+permutation CI
method), and 6 in `tests/pc_stable_adversarial.rs` (latent confounding as an
undisguised negative result, the bounded-`max_conditioning_set_size`
limitation via direct with/without comparison, `Inconclusive`-never-removes-
an-edge on genuinely independent data, a relabeling-invariance check,
determinism, and small-sample-count graceful degradation with 12 verified
"untestable candidate" warnings correctly propagated to the public result).
Every hand-derived expected `Cpdag` in every test — including the two
5-node/4-node integration cases — matched the implementation's actual output
on first run; none were adjusted after the fact to fit an implementation
bug.

### Compatibility

Purely additive. No existing public item's signature changed.
`examples/typed_causal_contract.rs` and
`examples/conditional_independence_benchmark.rs` untouched. The crate root's
docs are updated (five capabilities, not four) and one paragraph in the
"Causal interpretation" section is corrected: it no longer claims
equivalence-class discovery is out of scope for the *crate* — only for the
continuous optimizer specifically, which is what it always meant.

### Supported and unsupported claims

May claim: deterministic Markov-equivalence-class discovery via repeated
conditional-independence testing; an honestly three-way edge marking
(directed/undirected/absent) rather than a single guessed hypothesis DAG; a
documented, provably-complete (Meek 1995) orientation-propagation step for
the no-background-knowledge setting; conservative behavior under
`Inconclusive` or an untestable candidate (never an unjustified edge
removal); an honest warning, never a silent resolution, for a conflicting
v-structure demand.

Must **not** claim: that causal sufficiency or faithfulness has been
verified (both are assumed, not checked); that an undirected edge means no
causal relationship exists; that a bounded `max_conditioning_set_size` search
is complete; that this constructs a PAG or handles latent confounding in any
way; that any numerical causal effect has been identified or estimated.

### Known limitations / deferred

- No FCI / latent-confounding-robust discovery, hence no PAG — a future
  phase's explicit subject, not attempted here.
- Meek's rule 4 is not implemented (see "Design" above for why this is
  provably not a completeness gap in the no-background-knowledge setting
  this phase operates in).
- `max_conditioning_set_size` unbounded by default; a real, densely connected
  or high-dimensional variable set may need a caller-supplied bound, trading
  completeness for tractability — this phase does not choose a default bound
  on the caller's behalf.
- Effect identification, adjustment sets, invariance testing, interventions,
  counterfactuals, and experimental design remain out of scope — 5C.4
  onward, not to be started until this phase is merged and `master` is
  resynchronized.

## Phase 5C.4 — Estimate identifiable effects (backdoor adjustment)

**Status: Draft.** Branch `claude/scirust-srcc-robust-stats-6ue9xc`, restarted
from `origin/master` at `bfb6b8cf` (fresh master after PR #824 merged; this
phase's branch carries only this phase's commits). PR #840, merged at
`d371eebb` (with a follow-up documentation correction in PR #846). Additive to
`scirust-causal` (no existing public API changed). This phase implements
**identification by Pearl's backdoor criterion plus estimation by linear
covariate adjustment**. It does **not** implement front-door or
instrumental-variable identification, does **not** perform quantitative
sensitivity analysis, and does **not** relax the causal-sufficiency
assumption in any way.

This is the phase 5C.1's certificate layer was built for. That module's own
docs said, at the time: *"No phase in this crate yet produces a certificate
with `status = Identifiable` and a real estimate — that is later work."*
This is that work, and the structural rule 5C.1 installed (only
`Identifiable` may carry a numeric estimate) is now exercised end to end
rather than only unit-tested in isolation.

### Scientific scope — read before using this API

`estimate_effect_from_dag` answers a **two-part** question, and keeps the
parts separate:

1. *Identification* (pure graph reasoning, no data touched): does some set
   `Z` satisfy the backdoor criterion for (treatment, outcome) in the graph
   the caller supplied? If yes, `P(Y | do(X))` is a function of the
   observational distribution.
2. *Estimation* (data): under the **additional** assumption that `Y` is
   linear in `X` and `Z`, the coefficient of `X` in `Y ~ 1 + X + Z` is that
   effect.

Step 1 is a statement **about the supplied graph**, not a validation of it.
Every one of the following remains assumed and unchecked: that the graph is
correct; that there is no latent confounder (one absent from the graph is
invisible to a criterion evaluated *on* that graph); that positivity holds;
that the relationship is linear; that measurement is accurate. The phase's
headline adversarial result is a direct demonstration of the first two
failing together — see "Adversarial tests" below.

### Design

Three new modules plus the estimator:

- `d_separation.rs` — d-separation via the **ancestral-moralization**
  characterization (Lauritzen, Dawid, Larsen & Leimer 1990): `X ⟂d Y | Z`
  iff `X` and `Y` are separated by `Z` in the moral graph of `G` restricted
  to `An(X ∪ Y ∪ Z)`. This was chosen over a literal path-walk specifically
  because the collider-descendant clause ("a collider blocks unless it *or
  any descendant of it* is conditioned on") is a classic source of silent
  bugs in hand-rolled implementations; under moralization that clause is not
  written at all — it falls out of which nodes survive into the ancestral
  set — so it cannot be written wrong. Overlapping or out-of-range sets
  return the conservative "not separated" rather than a separation the
  definition does not license.
- `adjustment.rs` — the backdoor criterion. Condition 1 (no adjustment
  variable is a descendant of the treatment) is a direct descendant check;
  condition 2 (all backdoor paths blocked) uses the standard reduction: in
  the graph with **every edge out of the treatment deleted**, the only
  remaining paths from the treatment are exactly the backdoor paths, so
  condition 2 holds iff the treatment is d-separated from the outcome given
  `Z` in that mutilated graph. Also provides the canonical parents-of-
  treatment set and bounded enumeration of all *minimal* valid sets.
- `effect_estimation.rs` — the estimator and the certificate. Estimation
  uses the Frisch–Waugh–Lovell decomposition (residualize treatment and
  outcome on `[1, Z]`, then `β = Σx̃ỹ / Σx̃²`), which is *identical* to the
  full multiple regression's coefficient but reuses 5C.2's existing
  QR-with-SVD-rank-check residualizer — so a collinear adjustment set is a
  typed `RankDeficientConditioningSet` error rather than a silent
  pseudo-inverse. The standard error comes from the same residuals:
  `se = sqrt(σ̂² / Σx̃²)` with `σ̂² = RSS / (n − |Z| − 2)`.

### The three abstention paths

No estimate is produced — and, by the certificate layer's structural rule,
*cannot* be attached — in any of these cases:

| Situation | Status | Why abstain rather than report |
| --- | --- | --- |
| No valid backdoor set | `NotIdentifiable` | A regression would run fine and produce a confident number that is not a causal effect |
| CPDAG with an unoriented edge at the treatment | `EquivalenceClassOnly` | Members of the class disagree about the treatment's parents, so the class does not determine one effect |
| `n ≤ card(Z) + 2` | `Inconclusive` | The point estimate is arithmetically available but its uncertainty is not, and an effect with no quantifiable uncertainty is what this program exists not to report |

The CPDAG gate's condition — *every* edge incident to the treatment is
directed — is **sufficient, and not claimed necessary**: when it holds the
treatment's parent set is identical in every member DAG, so backdoor
adjustment by those parents gives the same answer for every member, and no
representative DAG need be chosen. Enumerating the full multiset of effects
across the class (the IDA approach), which would report a *range* instead of
abstaining, is deliberately out of scope and named below.

### Determinism contract

No RNG anywhere in this phase's own code. d-separation uses `BTreeSet`
adjacency and a fixed BFS order; adjustment-set enumeration is lexicographic
over an explicitly sorted candidate list; the estimator is QR least squares
(deterministic by construction) over a fixed row order inherited from 5C.2's
regime selection. Certificate fingerprints are therefore reproducible, and a
test asserts two identical runs produce byte-identical fingerprints.

### Tests

308 tests existed for `scirust-causal` before this phase (verified against
merged `master`). This phase adds **48**: 25 embedded unit tests (11
d-separation — chain/fork/collider, the collider-*descendant* case, the
M-structure, set-valued queries, overlapping and out-of-range guards; 14
backdoor — each condition in isolation, the canonical set, minimal-set
enumeration, and the `max_size` bound's honest incompleteness), 14 in
`tests/effect_estimation.rs` (recovering known coefficients of `+0.7`,
`+0.5`, `0.0`, `−0.6`; standard-error shrinkage with sample size; the
certificate mirroring the structured estimate exactly; fingerprint
reproducibility; JSON round-trip), and 9 in
`tests/effect_estimation_adversarial.rs`.

### Adversarial tests

- **Latent confounding (the headline negative result).** `U` confounds `X`
  and `Y` with a true effect of `0.7`, but `U` is in neither the data nor the
  graph. The backdoor criterion is *satisfied* for the graph supplied (`X`
  has no parents, so there is no backdoor path), the result is certified
  `Identifiable`, and the reported estimate is **1.497** — a bias of
  **121.5 standard errors**. A tight confidence interval around a badly
  wrong number, which is exactly the failure mode causal sufficiency's
  violation produces, demonstrated numerically rather than asserted in prose.
- Adjusting for a collider (M-structure) is refused, not silently biased.
- Adjusting for a mediator is refused by condition 1, and the correctly
  unadjusted query recovers the total effect `0.8 × 0.8 = 0.64`.
- A CPDAG with an unoriented edge at the treatment abstains; one fully
  oriented at the treatment estimates (both via real `PcStable` output, so
  this is a genuine 5C.3 → 5C.4 composition test, not a hand-built fixture).
- Exhausted degrees of freedom, a treatment fully determined by its
  adjustment set (`ZeroVariance`), and a collinear adjustment set
  (`RankDeficientConditioningSet`).
- A sweep asserting that **every** non-`Identifiable` path carries no
  estimate, no uncertainty, and no adjustment set.

### Benchmark

`examples/effect_estimation_benchmark.rs`: 13 scenarios, each checked
against an explicit oracle on status and (where a true coefficient exists)
on the estimate. The `latent_confounding` row prints its own bias and
bias-in-standard-errors, so the failure mode is visible in the output rather
than buried in a test name.

Oracle tolerances are expressed in **standard errors**, not absolute units.
An earlier draft used an absolute `0.02` bound, which is ~1.25 standard
errors at these sample sizes, and duly failed on a perfectly correct
estimate (`0.6746` against a truth of `0.7`, se `0.0160`). "Close to the
truth" is only meaningful relative to the estimator's own sampling noise;
the bound is now 4 standard errors throughout.

Run-twice SHA-256 (scientific stdout, nightly-2026-07-02, x86_64):
`7ac0dc767f76ef715f0282f51eda30411b6706dfb3d4c8be21912996fd14d93b`, verified
byte-identical across two runs and a debug/release build. The three
historical fingerprints — `industrial_protocol_demo` (`167c13de…`),
`conditional_independence_benchmark` (`c1449177…`), and
`pc_stable_benchmark` (`79e57e69…`) — were all independently reverified
unchanged.

### Compatibility

Purely additive. No existing public item's signature changed. Three
previously-private helpers were widened to `pub(crate)` for reuse rather
than duplicated (`skeleton_discovery::combinations`,
`partial_correlation::residualize`, and `conditional_independence`'s
`select_rows`/`extract_column`) — reuse keeps regime-selection and
rank-check semantics identical across phases instead of letting two copies
drift. The crate root's docs go from five capabilities to six, and the
out-of-scope paragraph is narrowed accordingly.

### Supported and unsupported claims

May claim: backdoor identification decided by a provably-correct
d-separation implementation; recovery of a known linear coefficient to
within sampling error; an honest three-way abstention discipline; a
certificate that names every assumption its estimate rests on and whose
fingerprint is reproducible.

Must **not** claim: that causal sufficiency, positivity, linearity, or graph
correctness have been *verified* (all are assumed); that an estimate is
robust to a latent confounder (it demonstrably is not); that abstention on a
CPDAG means no effect exists; that the `max_size` bound on minimal-set
enumeration proves no larger valid set exists.

### Known limitations / deferred

- Backdoor only. Front-door and instrumental-variable identification, which
  can identify effects the backdoor criterion cannot, are not implemented.
- Linear estimation only, matching 5C.2's own linear-association limitation.
- No IDA-style enumeration of the effect multiset across an equivalence
  class — the CPDAG path abstains where IDA would report a range.
- No quantitative sensitivity analysis (e.g. E-values, bias bounds under an
  assumed unmeasured confounder strength). The certificate carries a
  *qualitative* sensitivity note only; the adversarial test shows precisely
  why a quantitative one would be valuable, making it the most clearly
  motivated candidate for 5C.5 onward.
- Continuous variables only; binary/discrete treatments would need a
  different estimator.

## Phase 5C.5 — Quantify sensitivity to unmeasured confounding

**Status: Draft.** Branch `claude/scirust-srcc-robust-stats-6ue9xc`, restarted
from `origin/master` at `8f777520`. PR #850. Additive to `scirust-causal` (no
existing public API changed).

**This phase was inserted ahead of the planned invariance work, and the
roadmap renumbered accordingly** (invariance moves 5C.5 → 5C.6, and the rest
shift by one). The reason is empirical rather than aesthetic: 5C.4's own
adversarial test produced a latent confounder that made a certified estimate
wrong by 121 standard errors, and the only signal available at the time was a
prose sentence on the certificate. Building a quantitative answer to *how
strong would a confounder have to be?* was the most clearly motivated next
increment the program had, and it came from a measurement rather than a plan.

### Scientific scope — read before using this API

This module answers one question: **given a fitted linear model, how strong
would an omitted linear confounder have to be to move the estimate by a stated
fraction?** It is Cinelli & Hazlett (2020), *Making Sense of Sensitivity:
Extending Omitted Variable Bias*, JRSS-B 82(1).

It **quantifies a stated assumption**; it does **not** test whether a
confounder exists. No data can do that — a confounder absent from the data is
absent from every statistic computed on it. What this provides is a threshold
to compare against domain knowledge, not evidence about the world.

### Design

`src/sensitivity.rs`, entirely closed-form (no RNG, no simulation, no
re-fitting except in `benchmark_covariate`):

- **Robustness value** `RV_q = ½(√(f_q⁴ + 4f_q²) − f_q²)` with
  `f_q = q·|t|/√df` — the minimum share of residual variance a confounder must
  explain in **both** treatment and outcome to move the estimate by `100·q`
  percent. Bounded in `[0, 1)` by construction: as `f → ∞` it approaches but
  never reaches 1, since a confounder can never need to explain more than all
  the residual variance.
- **Partial R² of treatment with outcome** `= t²/(t² + df)` — the
  extreme-scenario benchmark.
- **Bias bound** `= se(β̂)·√df·√(R²_{Y~U|D,X}·R²_{D~U|X} / (1 − R²_{D~U|X}))`,
  with the adjusted standard error
  `se·√((1 − R²_Y)/(1 − R²_D))·√(df/(df − 1))`.
- **Covariate benchmarking** — expresses a hypothetical confounder in units of
  an *observed* covariate, via leave-one-out residualization reusing 5C.2's
  `residualize`/`pearson_correlation`. This is what makes `RV` actionable:
  "0.93" means little alone; "0.93, and the strongest covariate we measured
  reaches 0.04" means a great deal.

`analyze_sensitivity` **refuses a non-`Identifiable` estimate** with a typed
error. "How robust is this estimate?" presupposes an estimate, so the
abstention discipline of 5C.4 propagates: there is no way to run a sensitivity
analysis on something that was never identified.

### The central validation

The decisive test reconstructs 5C.4's latent-confounding scenario, then does
something a real analyst cannot: measures how strong the confounder *actually*
was, from a ground-truth dataset where `U` is present. Feeding those measured
strengths to the bias formula predicts the bias that was actually realised.

| Quantity | Value |
| --- | --- |
| Estimate (certified `Identifiable`) | 1.5163 |
| True effect | 0.7 |
| Actual bias | 0.8163 |
| Measured confounder strength (treatment) | 0.9007 |
| Measured confounder strength (outcome) | 0.4365 |
| **Predicted bias** | **0.8333** |
| **Relative error** | **2.09%** |

The measured strengths also match an independent analytic derivation from the
generating equations (`R²_D = 0.90`, `R²_Y ≈ 0.42`), so this is agreement
between three routes — the closed-form formula, the empirical measurement, and
the hand derivation — not a self-consistency check.

### The finding this phase records

**A high robustness value is not by itself reassurance.** On that same
scenario `RV₁ = 0.9335`, which reads as "you would need a confounder
explaining 93% of both residual variances to overturn this" — apparently very
robust. The confounder actually present explained **90.1%** of the treatment's
residual variance. It was very nearly that strong, and it was real.

`RV` is a threshold, not a verdict. It is only informative when read against
what is plausible in the domain, which is exactly what `benchmark_covariate`
exists to supply. Both halves are pinned down by tests.

### A tolerance correction worth recording

An early draft of the decisive test asserted that the adjusted range must
*contain* the true effect. It failed by ~2% of the bias on one seed. The cause
is not a defect: the Cinelli–Hazlett bias is **exact** for a confounder of
exactly the given partial R² values — it is a "bound" only over the unknown
*direction*, and is not conservative in its inputs. Those inputs are measured
from finite data, so the prediction can land marginally under the realised
bias. The assertion now allows a shortfall of up to 5% of the bias bound, with
the reason stated in the test, rather than claiming a guarantee the method does
not make.

### Determinism contract

No RNG anywhere in this phase's code; every reported quantity is a closed-form
function of `(β̂, se, df)` plus, for benchmarking, deterministic QR
residualization. Verified byte-identical across two runs and a debug/release
build.

### Tests

338 tests existed for `scirust-causal` before this phase. This phase adds
**18**: 7 embedded unit tests (RV bounded/monotone; the hand-computed
`f = 1 ⇒ RV = ½(√5 − 1) = 1/φ` case; scenario validation; a null confounder
inducing exactly zero bias; bias monotone in strength; adjusted standard error
moving in both documented directions; and an internal cross-check that a
confounder at exactly `RV` induces a bias equal to the estimate — validating
the RV formula against the *independent* bias formula), and 11 integration
tests including the decisive recovery above, the RV-misleads finding, weak- vs
strong-evidence RV ordering, `RV_q` monotonicity in `q`, irrelevant-covariate
benchmarking, and the refusal on non-`Identifiable` input.

### Benchmark

`examples/sensitivity_benchmark.rs`, 5 scenario groups, oracle-checked, with
the latent-confounder row printing its own measured strengths, predicted bias,
actual bias and relative error so the central claim is visible in the output.

Run-twice SHA-256:
`1bc59a1d58facd07f3150ba94cb9e2d7762fc5aaf9f918e730ea6a9e40b52ea7`.
All four prior fingerprints reverified unchanged: `industrial_protocol_demo`
(`167c13de…`), `conditional_independence_benchmark` (`c1449177…`),
`pc_stable_benchmark` (`79e57e69…`), `effect_estimation_benchmark`
(`7ac0dc76…`).

### Compatibility

Purely additive. No existing public item's signature changed; no new
`pub(crate)` widenings were needed (5C.4 already exposed `residualize` and
`pearson_correlation`). Crate root goes from six capabilities to seven, and
the out-of-scope paragraph now distinguishes *quantifying* a latent
confounder's potential damage (done) from *detecting or removing* one (still
out of scope).

### Supported and unsupported claims

May claim: a closed-form, deterministic robustness value and bias bound for a
linear backdoor-adjusted estimate; agreement with a known confounder's
realised bias to ~2%; expression of a hypothetical confounder in units of an
observed covariate; refusal to analyse an unidentified effect.

Must **not** claim: that a confounder has been detected, excluded, or
corrected for; that a high `RV` means an estimate is trustworthy; that the
bound covers nonlinear confounding, measurement error, or model
misspecification other than an omitted linear term.

### Known limitations / deferred

- Linear omitted-variable bias only. Nonlinear confounding, effect
  modification by the confounder, and measurement error are out of scope.
- No E-values. The E-value is defined on a risk-ratio scale and would need a
  standardized-mean-difference conversion to apply here — an approximation
  layer offering no additional rigour over a framework that is *exact* for the
  linear model 5C.4 actually produces. Deliberately omitted rather than
  included for coverage's sake.
- `benchmark_covariate` handles one covariate at a time; the "k times as
  strong as covariate j" multiplier form of Cinelli–Hazlett is not
  implemented.
- No formal significance-adjusted robustness value (`RV_{q,α}`).
