# Scientific capability gap audit

Status: decision record after repository audit

Baseline: `master` at `af37d0dfd76685a96df165912b83557823620d48`

This document records the decision gate for three candidate scientific-computing capabilities investigated after the ElasticXxx literature work. The candidates were investigation targets, not implementation commitments. ElasticXxx-specific semantics are deliberately excluded, and ElasticXxx is not introduced as a runtime dependency.

## Decision summary

| Candidate | Repository finding | Decision |
| --- | --- | --- |
| Generic LP / ILP / MILP | **TRUE GAP** in generic modelling, with **EXTERNAL LIBRARY STRONGER** | Do not implement a MIP solver in SciRust. Reconsider an optional adapter only for a concrete scientific consumer. |
| Advanced predictive control | **PARTIALLY EXISTS** | Do not add NMPC, explicit MPC, or multi-rate MPC without an independent benchmark and solver/model requirements. Do not duplicate the existing linear MPC. |
| Right-censored survival analysis | **TRUE GAP**, scientifically justified | Foundational support implemented and merged by PR #1335. Defer Cox PH until lower-level survival abstractions and validation mature. |

## Audit method

The audit inspected implementation and public APIs rather than relying only on keyword search. The relevant surfaces included:

- `scirust-solvers` continuous, nonlinear, combinatorial, and linear-algebra solvers;
- `scirust-control` LQR, QP, PID, and MPC;
- `scirust-estimation` state estimation, recursive least squares, and MIMO FIR identification;
- `scirust-process` process models;
- `scirust-bms` thermal monitoring;
- `scirust-stats` distributions, inference, and robust statistics;
- `scirust-reliability` reliability and functional-safety probability models;
- sparse and matrix support used by the solver stack;
- repository searches for equivalent APIs under other names and for external-solver integration.

## 1. Generic LP / ILP / MILP

### Existing SciRust capability

`scirust-solvers` already provides substantial optimization machinery, including BFGS, gradient descent, Nelder-Mead, spectral projected gradient under box constraints, nonlinear root/system solvers, dense and sparse/matrix-free linear algebra, and specialized combinatorial algorithms. Its combinatorial layer includes exact 0/1 selection under a budget and deterministic submodular selection.

Those specialized algorithms are useful scientific primitives, but they are not a reusable LP/MILP modelling layer. The audited public API does not expose the usual generic model of decision variables, linear objective, arbitrary linear constraints, integrality declarations, solver status, and backend-independent solution access.

### Why a SciRust MIP solver is not justified

A production MIP solver is not simply a simplex implementation plus branch-and-bound. Competitive solvers contain presolve, numerical scaling, basis management, cut generation, heuristics, node selection, conflict processing, and extensive numerical safeguards. Reimplementing that stack would create a large maintenance and validation burden without a demonstrated scientific advantage for SciRust.

Mature Rust-facing alternatives already separate modelling from solver implementation. In particular:

- `good_lp` describes itself as a mixed-integer linear-programming modeller and supports continuous and integer variables with multiple solver backends, including HiGHS and SCIP: <https://github.com/rust-or/good_lp>;
- HiGHS provides LP/MIP algorithms and has a Rust-facing crate used by `good_lp`: <https://github.com/rust-or/highs>;
- SCIP has a safe Rust interface through `russcip`: <https://github.com/scipopt/russcip>.

### Decision

**No solver code is added.**

An optional SciRust adapter may be reconsidered only when all of the following are true:

1. a concrete scientific SciRust crate needs a generic LP/MILP model rather than a specialized algorithm;
2. the required problem classes and status/duality semantics are written down first;
3. backend licensing, native-build requirements, MSRV, deterministic-test behaviour, and supported targets are explicitly evaluated;
4. the adapter does not pretend that backend-dependent numerical or MIP search behaviour is bit-deterministic;
5. there is a reference problem suite that can be cross-validated against the selected backend independently of ElasticXxx.

Until then, adding a generic modelling abstraction would be speculative API surface with no validated consumer.

## 2. Advanced predictive control

### Existing SciRust capability

`scirust-control` already contains a `LinearMpc` implementation for discrete linear state-space dynamics

`x[k+1] = A x[k] + B u[k]`

with a finite-horizon quadratic objective and hard box input constraints. The problem is condensed and solved through the crate's box-QP implementation. LQR and PID are separate existing controllers.

The current QP helper is intentionally narrower than a general constrained QP solver: it solves convex box-constrained problems with projected-gradient iterations. This is adequate for the current `LinearMpc` contract but is not a credible nonlinear-programming foundation by itself.

`scirust-estimation` already supplies Kalman-family estimators and online MIMO FIR recursive least-squares identification. Therefore, adding a second generic "MPC" implementation would duplicate existing capability rather than close the advanced-control gap.

### Candidate advanced directions

The audit considered four possible missing layers:

- nonlinear MPC (NMPC);
- explicit MPC;
- multi-rate MPC;
- system-identification-to-MPC integration.

None currently has enough repository evidence to justify implementation.

For NMPC in particular, a reusable implementation requires choices that cannot be made honestly without a target scientific problem: nonlinear dynamics representation, discretization/integration, single versus multiple shooting or collocation, Jacobian/gradient strategy, state/input/path constraints, nonlinear solver, warm starts, convergence reporting, and failure behaviour. Existing external work such as Optimization Engine illustrates that NMPC is a parametric non-convex optimization problem rather than a small extension of a box-QP: <https://github.com/alphaville/optimization-engine>.

The process models inspected during this audit do not yet provide a sufficiently general nonlinear dynamic benchmark. For example, current CSTR and batch-reactor utilities are primarily steady-state or closed-form engineering calculations rather than reusable nonlinear discrete-time state models for controller validation. Likewise, the BMS thermal guard is a monitoring primitive, not a predictive plant model.

### Decision

**Do not add NMPC, explicit MPC, or multi-rate MPC in this audit.**

System-identification-to-MPC integration remains the most plausible direction to investigate next because SciRust already has MIMO FIR RLS and linear MPC. It still requires a separate decision gate. Before code is justified, the repository needs:

1. an independent scientific benchmark, not an ElasticXxx workload;
2. a precise identified-model contract and uncertainty/validation semantics;
3. a defined conversion or prediction interface between the identified model and MPC;
4. a reference controller/result against which closed-loop behaviour can be checked;
5. deterministic tests that distinguish modelling error from optimizer convergence error.

Without those items, a bridge would merely couple two APIs without establishing scientific validity.

## 3. Right-censored survival analysis

### Audit result

Before PR #1335, `scirust-stats` covered probability distributions, descriptive and robust statistics, and classical hypothesis tests, while `scirust-reliability` covered SIL/PFD/PFH, MooN architectures, and simple Markov availability. Neither surface represented right-censored observations or exposed foundational non-parametric survival estimators.

This was a genuine cross-domain scientific gap: censored time-to-event data occur in reliability studies as well as biomedical and other lifetime analyses.

### Implemented foundation

PR #1335 (`feat(stats): add right-censored survival foundations`) was merged into `master` as `af37d0dfd76685a96df165912b83557823620d48`. It adds:

- a validated right-censored observation type;
- Kaplan-Meier product-limit estimation;
- Nelson-Aalen cumulative-hazard estimation;
- the two-sample Mantel log-rank test with hypergeometric variance;
- explicit deterministic handling of event/censor ties;
- analytical and edge-case tests;
- public and prelude exports;
- no external runtime dependency and no ElasticXxx dependency.

The implementation references the foundational literature:

- E. L. Kaplan and P. Meier, "Nonparametric Estimation from Incomplete Observations", *JASA* 53(282), 1958, DOI 10.1080/01621459.1958.10501452;
- W. Nelson, "Theory and Applications of Hazard Plotting for Censored Failure Data", *Technometrics* 14(4), 1972, DOI 10.1080/00401706.1972.10488991;
- N. Mantel, "Evaluation of survival data and two new rank order statistics arising in its consideration", *Cancer Chemotherapy Reports* 50(3), 1966, PMID 5910392.

### Cox proportional hazards gate

Cox PH is deliberately **not** part of the foundational PR. A future regression layer should not be added until its data/model abstractions, tie treatment, optimization method, convergence diagnostics, and validation against published/reference datasets are specified independently. The existence of Kaplan-Meier/Nelson-Aalen/log-rank is a prerequisite, not sufficient justification by itself.

## Resulting policy

This audit closes the three candidate questions without turning a literature-derived wish list into code automatically:

- SciRust should reuse mature external optimization engines when a generic MIP backend is actually required rather than attempting to become a MIP solver project.
- SciRust should extend predictive control only around a demonstrated scientific control problem with reference validation, while preserving the existing linear MPC API.
- SciRust now has a small, coherent right-censored survival foundation; higher-level survival regression remains gated on explicit mathematical and numerical validation.

Any future implementation proposal in these areas should cite this decision record and state what new evidence changes the relevant gate.
