# Nonlocal research harvest map

Status: design contract for progressive de-specialization and reuse.

Audited target head: `Memorithm/scirust@8cd016e3103851644e32950c6828cd1d54be5ded` (`master`).

Audited research head: `Memorithm/nonlocal-relativity-v2@8cafd475244ff8490c6306d07f7424d83a664484` (`main`).

This document records what may be generalized from `nonlocal-relativity-v2`, what is already owned by SciRust, and what must remain relativity-specific. It is deliberately a transfer map, not a claim that relativistic formulas are useful outside relativity.

## Scientific and architectural boundary

The reusable result of `nonlocal-relativity-v2` is an engineering pattern for deterministic systems whose effective behavior depends on structured history:

```text
current state
+ retained historical values
+ true historical positions
+ optional representation transform
+ optional contextual weighting
+ explicit history kernel
+ explicit approximation policy
+ adaptive resource/error control
+ independently classified evidence
-> next state or decision
```

The research repository does **not** establish that Schwarzschild, Kerr, curvature, proper-time, parallel-transport, or other general-relativistic formulas improve machine-learning attention or enterprise memory. Those formulas remain in their scientific domain.

Ownership after harvest is intentionally acyclic:

- **SciRust** owns domain-neutral history storage/view contracts, true historical positions, explicit reference-vs-approximation classification, generic transform/weight/kernel interfaces where justified, generic adaptive-control primitives where they have multiple concrete consumers, and scientific evidence classification.
- **FLAT-ATTENTION** owns attention semantics, semantic identities, scalar/reference attention behavior, KV/history execution, Kernel IR, candidate qualification, hardware evidence, and any experimental nonlocal/recurrent attention rule.
- **CCOS Enterprise** owns authentication, RBAC, tenant isolation, policy, quotas, budgets, governed retention, persistence/restore, audit, and Enterprise memory policy. A memory score is never authority.

SciRust must not depend on FLAT-ATTENTION or CCOS Enterprise in order to expose generic history machinery.

## Provenance map

The research history is large, so provenance is clustered around conceptual transitions rather than replaying every commit linearly.

| Transition | Canonical provenance | What it established |
| --- | --- | --- |
| Initial worldline memory / complete and bounded history foundations | research commits `2950992`, `dbc548f` (referenced by PR #1) | Fixed-step fractional-memory worldline foundation and the pre-transport history architecture. |
| Typed history source point, transport, proper-time mode, modulation | PR #1; commits `45526b8`, `37c1f62`; merged as `cddac38` | `HistoryEntry`, `HistoryBackend::push_entry`, `HistoryTransport`, discrete transport, `HistoryModulator`, identity/bypass paths. |
| Independent exact flat transport oracle | PR #2; commit `a5479e3` | Exact cylindrical-Minkowski reference and convergence comparison against discrete transport. |
| True non-uniform Caputo operator | PR #3 | `scirust-fractional::caputo_l1_nonuniform`; actual sample positions are consumed instead of reconstructed uniform positions. |
| Curved transport oracle and live adaptive stepping | PR #4 | Exact Schwarzschild circular-orbit transport oracle; embedded adaptive Heun-Euler; second background/modulator family. |
| Adaptive composition with history transport/modulation | PR #5 | Policy composition and bit-identical identity special case; Kerr remains explicitly numerical/relativistic. |
| Non-uniform `MemoryLaw` and adaptive stepper composition | PR #6 | Memory evaluation from retained `HistoryEntry::parameter` and explicit rejection-budget behavior. |
| Adaptive hardening and true accumulated parameter | PR #7 | Shared scaled error controller; separate rejection/minimum-step exhaustion; `StepperContext::current_parameter` replaces `step_index * step` reconstruction. |
| Evidence taxonomy and negative-result discipline | PR #8, `paper/nonlocal-relativity-v2.md` | Explicit separation of exact analytic result, numerical oracle, fine-grid reference, self-convergence, regression, and physical validation; negative history-retention result retained. |
| Bounded-history qualification | PR #10 | Deterministic bounded-vs-complete experiment; exact equality once the bounded window covers all samples. |
| Platform architecture audit | PR #12 | Canonical crate/subsystem map and explicit anti-duplication findings. |
| Consolidated non-uniform memory kernel | PR #13; commit `2e329435080eda515ac688e309f8377c1de9839c` | Moved duplicate non-uniform Caputo memory builders into one internal implementation with unchanged bit-identity goldens. |
| Reusable GR geometry transport/tetrad/bitensor work | PRs #14, #18, #19, #21 | Established-GR numerical geometry infrastructure; useful validation methodology, but not generic history semantics. |
| Robust statistics synchronization | PR #35; research head merge `8cafd475244ff8490c6306d07f7424d83a664484` | `scirust-stats::robust` was imported **from upstream SciRust into the research fork**, so it is not a harvest target. |

Canonical modules at the audited research head include:

- `scirust-fractional` for uniform and non-uniform Caputo operators;
- `scirust-nonlocal-relativity/src/lib.rs` for `HistoryBackend`, complete/bounded history, `MemoryLaw`, and stepper contracts;
- `scirust-nonlocal-relativity/src/transport.rs` for `HistoryEntry` and discrete history transport;
- `scirust-nonlocal-relativity/src/nonuniform_kernel.rs` and `nonuniform_memory.rs` for non-uniform history evaluation;
- `scirust-nonlocal-relativity/src/modulation.rs` for history modulation;
- `scirust-nonlocal-relativity/src/adaptive_control.rs`, `adaptive.rs`, and `adaptive_stepper.rs` for adaptive control;
- the deterministic experiment suite and technical paper for convergence, approximation, and evidence methodology.

## Current SciRust state

The audited SciRust `master` already contains the imported numerical/relativistic stack. In particular, `scirust-nonlocal-relativity` currently exposes:

- `HistoryApproximation::{Exact, Approximate}`;
- `HistoryDiagnostics`;
- `HistoryBackend<const D: usize>`;
- `CompleteUniformHistory`;
- `BoundedShortMemoryHistory`;
- `HistoryEntry<D> { coordinates, velocity, parameter }`;
- `HistoryTransport` and `IdentityHistoryTransport`;
- relativity-specific transport and modulators;
- uniform and non-uniform Caputo memory laws;
- `StepperContext::current_parameter`, explicitly documented as the true accumulated parameter that must replace reconstruction from a step index under non-uniform spacing.

Those are valuable implementations, but the storage contracts are still coupled to `[f64; D]`, `WorldlineState`, `Connection`, coordinates, velocity, and relativistic error types. The next step is therefore **de-specialization**, not another source import.

`scirust-fractional` already owns the generic Caputo operators. `scirust-stats` already owns robust statistics. These must not be duplicated into a new history crate.

## Classification

The classification values are normative for later harvest PRs:

- **TRANSFER GENERIC** — extract a domain-neutral mechanism into SciRust generic infrastructure.
- **ADAPT** — preserve the design idea but redesign the API around domain-neutral types; the source implementation itself is too specialized to move verbatim.
- **REFERENCE ONLY** — keep as validation/provenance/methodology; no new generic implementation is needed.
- **RELATIVITY ONLY** — remain in relativity/nonlocal-relativity crates.
- **DO NOT TRANSFER** — explicitly prohibited from downstream generalization.

| Construct | Classification | Reason / required action |
| --- | --- | --- |
| Uniform Caputo | **REFERENCE ONLY** | Already a generic `scirust-fractional` operator. History infrastructure may adapt to it, but must not duplicate it. |
| Non-uniform Caputo | **REFERENCE ONLY** | Already generic in `scirust-fractional` via PR #3. Preserve the actual-position contract and convergence tests; do not create a competing evaluator. |
| Full/complete history | **TRANSFER GENERIC** | The reference-history concept is domain-neutral. Extract complete retention/storage without `[f64; D]`, coordinates, velocities, or `Connection`. |
| Bounded history | **TRANSFER GENERIC** | Explicit bounded retention is domain-neutral. It must remain explicitly approximate and never silently replace complete history. |
| Typed historical position | **TRANSFER GENERIC** | `HistoryEntry::parameter` and PR #7 demonstrate that true position is semantic data under non-uniform spacing. Generic form should be `HistoryEntry<Value, Position>`. |
| History retention | **TRANSFER GENERIC** | Retention/view/accounting can be generic. Product policy (tenant quotas, authorization, attention semantics) must remain downstream. |
| History transform | **ADAPT** | General reusable idea: old value plus source metadata/context -> comparable value now. GR parallel transport remains a specialization. Provide an exact identity transform. |
| History modulation / weighting | **ADAPT** | General reusable idea: contextual weight independent of age/kernel. Curvature and field-invariant formulas remain relativity-only. Generic identity must return exact multiplicative identity. |
| Adaptive error/resource controller | **ADAPT** | PR #7 contains a reusable propose/evaluate/normalize/accept-or-retry/budget/clamp pattern, but SciRust `scirust-sim` must be audited first and reused rather than duplicated. |
| Exact transport oracles | **RELATIVITY ONLY** | Cylindrical-Minkowski and Schwarzschild circular-orbit transport are domain-specific exact/numerical validation references. Their *oracle discipline* is reusable; the formulas are not. |
| Convergence studies | **REFERENCE ONLY** | Reuse methodology and evidence labels. Do not copy experiment-specific equations into generic crates. |
| Negative-result retention | **TRANSFER GENERIC** | The evidence discipline is domain-neutral: failed/rejected hypotheses are first-class evidence. Integrate with existing SciRust provenance/evidence infrastructure rather than a parallel registry. |
| Robust statistics | **REFERENCE ONLY** | PR #35 proves the direction was upstream SciRust -> research fork. Continue to use `scirust-stats`; do not re-harvest it. |
| GR metrics in general | **RELATIVITY ONLY** | Metrics, connections, curvature, geodesics and associated invariants belong to `scirust-relativity`. |
| Schwarzschild / Reissner-Nordstrom / Kerr | **RELATIVITY ONLY** | Established/specialized GR backgrounds; no downstream ML/Enterprise analogy is evidence. |
| BSSN / ADM / numerical-relativity formulations | **RELATIVITY ONLY** | Scientific ownership remains in relativity/numerical-relativity infrastructure. They are unrelated to generic retention contracts. |
| Proper time | **RELATIVITY ONLY** | Proper-time parameterization is physical/geometric semantics. The generic lesson is only that the real position/parameter must be stored explicitly. |
| Tensor/geometric infrastructure | **REFERENCE ONLY** | SciRust already owns tensor representation and geometry infrastructure. Reuse it in its native domains; do not create history-specific copies. |
| Relativity-specific memory law in ML/Enterprise | **DO NOT TRANSFER** | No evidence supports deriving an attention or Enterprise retention law from GR analogy. Any downstream rule needs an independent mathematical/product definition and evaluation. |
| Reconstructing position as `index * constant_step` without a validated uniform-spacing invariant | **DO NOT TRANSFER** | PR #7 fixed this exact failure mode. Non-uniform retained histories must carry their original position. |
| Treating bounded/sampled/compressed history as exact | **DO NOT TRANSFER** | Complete retained history is the numerical reference in this program; reductions are explicit approximations. |
| Treating self-convergence/regression/benchmark results as independent proof | **DO NOT TRANSFER** | These evidence classes answer different questions and must remain distinct. |

## Generic foundation contract

The smallest coherent generic foundation should preserve the following semantics without importing general relativity:

```text
HistoryEntry<Value, Position> {
    value,
    position,
}

HistoryBackend / HistoryView
CompleteHistory
BoundedHistory
RetentionPolicy (only storage/selection semantics)
retained sample count
original/logical position
explicit reference-vs-approximation status
```

The implementation must not assume uniform spacing. Duplicate-position behavior and ordering guarantees must be explicit and tested rather than inferred.

The generic foundation must not contain:

- Christoffel symbols, metrics, curvature, tetrads, proper time, or GR coordinates;
- FLAT semantic IDs, KV layout/routing, RoPE or EPG policy;
- CCOS tenant identity, RBAC, authorization, quotas, persistence policy, or audit authority.

## Behavior-preservation gate for the relativity adapter

Generalization is acceptable only if the existing nonlocal-relativity behavior remains unchanged on behavior-preserving paths.

Required evidence for the adapter work:

1. Existing complete-history reference behavior remains unchanged.
2. Existing bounded-history behavior remains unchanged for the same retained entries.
3. Identity/no-transform paths preserve values bit-for-bit where the current implementation already does.
4. `beta = 0` / disabled modulation continues to reproduce its baseline exactly where current goldens require it.
5. Non-uniform evaluation continues to use recorded `HistoryEntry::parameter` values, never reconstructed indices.
6. A bounded window covering the entire available history agrees exactly with complete history where the source experiment/test already establishes exact equality.
7. Existing exact transport oracles, adaptive goldens, convergence tests and non-uniform/uniform cross-checks remain green unless a later PR explicitly changes a scientific contract with new evidence.

Moving a function verbatim is not by itself proof of semantic equivalence; the existing golden/oracle tests remain authoritative.

## History-kernel direction

After the storage foundation is stable, a generic evaluation contract may be introduced only at the smallest useful level. It should consume a history view carrying true positions and explicit approximation status. Caputo is an implementation/adapter through `scirust-fractional`, not the definition of all history kernels.

The later kernel PR must preserve these reference checks:

- uniform and non-uniform Caputo agreement on a genuinely uniform grid to the established numerical tolerance;
- analytical Caputo cases already present in `scirust-fractional`;
- the non-uniform linear-function oracle where currently applicable;
- complete and bounded histories agreeing when bounded retention covers every sample.

No complexity improvement may be claimed unless an algorithm implementing it is actually present and measured.

## Transform and weighting direction

A generic transform represents only:

```text
historical entry + current context -> transformed historical value
```

Its identity implementation must be an exact no-op. Missing metadata required by a specialized transform must fail closed rather than inventing a source position.

A generic history weight represents only:

```text
historical entry + current context -> validated weight
```

Its identity implementation must return the exact multiplicative identity. Non-finite weights must be rejected. Schwarzschild/Kerr/Reissner-Nordstrom-specific quantities remain in the relativity layer.

## Adaptive-control direction

Before extracting anything, audit `scirust-sim` and any existing generic observation/control contracts. Reuse existing step-size/error-control infrastructure where semantically compatible.

The reusable pattern from PR #7 is:

```text
PROPOSE
-> EVALUATE
-> NORMALIZE ERROR OR UTILITY
-> ACCEPT / RETRY / REJECT
-> APPLY EXPLICIT BUDGET
-> CLAMP TO RESOURCE/TARGET BOUNDS
-> COMMIT
```

Important retained semantics are deterministic decisions, explicit budget exhaustion, distinct failure reasons, and no silent acceptance after exceeding the requested error/budget contract.

Do not create a generic resource controller until there is a concrete second consumer beyond numerical integration.

## Evidence direction

The technical paper and experiment suite distinguish evidence that must not be collapsed into one boolean such as `validated`.

The target taxonomy should be integrated with existing SciRust provenance/evidence facilities and should be able to distinguish at least:

- exact mathematical result;
- validated numerical implementation;
- numerical approximation;
- phenomenological model;
- speculative model;
- empirical validation;

and, where useful, oracle strength such as:

- exact oracle;
- independent numerical oracle;
- fine-grid reference;
- self-convergence;
- regression golden;
- hardware measurement.

Self-convergence does not establish physical correctness. Regression equality is not an independent oracle. A hardware benchmark is evidence only for its recorded environment. Negative results must remain recordable as first-class evidence.

## Downstream transfer rules

### FLAT-ATTENTION

The reusable architecture may support a research-only nonlocal/recurrent/multiscale attention program, but every changed mathematical rule is a distinct semantic contract. `StandardSoftmax` must never be silently substituted with a recurrent/nonlocal semantic under resource pressure.

Historical KV/state retained after sampling must preserve original positions. Positional/representation transforms, history weighting, retention policy and the history kernel are separate concerns. CPU/full-history reference work must precede any WGPU candidate or production routing.

### CCOS Enterprise

Generic history primitives may support governed memory qualification, but Enterprise remains authoritative for tenant scope, authentication, RBAC, quotas, budgets, persistence and audit. Similarity/relevance/history weight may influence retrieval or retention but can never authorize an action.

Reference/full retention is an experimental oracle for measuring information loss; it is not automatically a production forever-retention policy. Any compaction must preserve logical position and provenance needed by the restore/audit contract.

## Negative result preserved from the source program

The technical paper associated with PR #8 explicitly records that retaining denser history did **not** materially improve the investigated endpoint while increasing cost. PR #10 separately shows a bounded-history experiment in which error decreases as the window grows and becomes exactly zero once the window covers all available samples.

These observations are not contradictory: they concern measured configurations and metrics. Together they motivate the generic rule that retained sample count is not itself a quality metric. Downstream qualification must compare a reduced-history strategy against an explicit reference using a task-relevant error/utility measure.

This negative result must not be removed merely because later FLAT or CCOS experiments pursue richer retention strategies.

## Planned SciRust PR sequence

This document authorizes only the direction, not unreviewed API details. The intended small-PR sequence is:

1. generic history foundation and tests;
2. behavior-preserving `scirust-nonlocal-relativity` adapter if separation is cleaner as its own PR;
3. generic history-kernel contract with `scirust-fractional` adapters;
4. generic identity transform and weighting contracts, with relativity specializations remaining local;
5. adaptive-control integration only after `scirust-sim` audit demonstrates a real missing generic abstraction;
6. evidence taxonomy integrated with the existing provenance/evidence model.

Every implementation PR must record whether source code was moved verbatim, adapted, or independently reimplemented, and must cite the source module/PR/commit used for provenance.

## Non-goals

This harvest does not:

- claim new physics;
- modify Einstein's equations;
- claim a GR-derived ML memory law;
- claim nonlocal attention improves a model before real-model evaluation;
- define CCOS authorization from memory relevance;
- promote an approximation to reference status;
- erase failed experiments;
- weaken numerical tolerances, security boundaries, semantic identities, or CI gates to simplify integration.

The success criterion is narrower: preserve validated research history while moving only domain-neutral mechanisms to their correct architectural owner.