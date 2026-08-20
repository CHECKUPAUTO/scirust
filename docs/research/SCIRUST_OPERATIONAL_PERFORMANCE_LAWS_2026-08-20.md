# Operational Performance Laws — SciRust Research Note

Date: 2026-08-20

## Motivation

A review of DS2 (OSDI 2018) for the ElasticXxx research program highlighted the distinction between observed throughput and sustainable processing capacity. A broader review of Denning & Buzen, *The Operational Analysis of Queueing Network Models* (ACM Computing Surveys 10(3), 1978), showed that a reusable scientific core can be formulated in terms of directly measurable operational quantities rather than an Elastic-specific autoscaling model.

SciRust already contains a deterministic discrete-event `MM1Queue` simulator in `scirust-sim::stochastic`, including tests against the classical M/M/1 formulas and Little's law. The missing reusable layer was therefore narrower than "queueing theory": basic operational identities and deterministic service-demand/bottleneck analysis were used as implicit oracles but not exposed as API.

## Scope added

`scirust-sim::operational` provides:

- utilization law `U = X·S` for a single unit-capacity service center;
- Little's law in `N = X·R` and `R = N/X` forms;
- forced-flow law `X_i = V_i·X_0`;
- interactive response-time relation `R = M/X - Z`;
- `ServiceDemand` with `D = V·S`;
- deterministic bottleneck analysis using the maximum service demand;
- saturation-throughput bound `1 / max(D_i)`;
- sum-of-demands no-wait service-time baseline.

## Deliberate non-goals

This change does **not** add:

- an ElasticXxx planner;
- autoscaling policy;
- a general queueing-network solver;
- M/M/c, M/G/1, Jackson/Gordon-Newell/BCMP solvers;
- parameter fitting;
- confidence intervals;
- simulation of arbitrary queueing networks;
- claims that operational prediction assumptions always hold.

## Evidence discipline

The laws are mathematical relationships among declared/measured quantities. Predictive uses still require callers to justify assumptions such as flow balance or invariance of service demands under a proposed change.

The API therefore does not hide those assumptions behind an autonomous "capacity predictor".

## Validation

Unit tests include:

- exact low-dimensional identities;
- malformed-input rejection;
- reproduction of the Denning–Buzen example with service demands `1.00`, `0.88`, and `0.32`, total `2.20`, CPU/index 0 as bottleneck, and saturation throughput `1.0` in the paper's normalized units.

The existing M/M/1 simulation remains an independent stochastic cross-check for future work.

## Architectural boundary

SciRust remains a general scientific R&D platform. ElasticXxx has no SciRust runtime dependency. ElasticXxx may use SciRust during research to formulate and test performance/capacity models, after which any selected production mechanism is implemented autonomously in ElasticXxx or its target runtime.
