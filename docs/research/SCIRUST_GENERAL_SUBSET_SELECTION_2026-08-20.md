# SciRust General Subset-Selection Primitives — 2026-08-20

## Status

Implemented on `master` as general scientific optimization primitives in `scirust-solvers`.

These capabilities were revealed while studying resource-management and KV-cache literature, but **they are not KV-specific and do not depend on ElasticXxx**.

## 1. Exact additive selection under a budget

Module:

```text
scirust-solvers/src/combinatorial/budgeted_selection.rs
```

Problem:

```text
maximize   Σ utility_i x_i
subject to Σ cost_i x_i ≤ budget
           x_i ∈ {0,1}
```

Properties:

- exact for supplied integer/fixed-point costs and utilities;
- deterministic tie-breaking;
- sparse Pareto-frontier dynamic program;
- memory does not scale directly with the numeric budget value;
- returns explicit exactness/work metadata.

Typical scientific uses include bounded experimental selection, cache-object selection, feature selection with additive utility, task portfolios, sensor subsets with additive scores, and small exact baselines for heuristic evaluation.

## 2. Greedy monotone-submodular selection under a cardinality budget

Module:

```text
scirust-solvers/src/combinatorial/submodular.rs
```

Problem:

```text
maximize   F(S)
subject to |S| ≤ k
```

Caller contract:

- `F(∅)=0` (normalized);
- `F` is monotone;
- `F` is submodular;
- supplied marginal gains are exact and non-negative.

Under those assumptions, classical greedy provides the standard `(1 - 1/e)` approximation guarantee for a cardinality constraint.

### Honesty boundary

SciRust **does not claim to prove** that a black-box callback is normalized, monotone or submodular by sampling it. The result certificate records the theorem whose assumptions the caller has opted into. This is an algorithmic guarantee conditional on a mathematical contract, not an automatic proof of the contract.

The implementation is deterministic: candidates are evaluated in ascending index order and equal marginal gains select the lowest index.

Typical uses include coverage, summarization, diversity-aware selection, sensor placement, experimental design, caching with diminishing returns, and feature-set selection.

## 3. Why both primitives are needed

Additive utility assumes:

```text
U(S) = Σ U(i)
```

Submodular utility allows diminishing returns:

```text
Δ(i | A) ≥ Δ(i | B)    when A ⊆ B
```

The latter captures redundancy among selected elements. Neither model is universally correct; callers should use the strongest structure justified by the scientific problem.

## 4. Literature trigger

The general need became clear while comparing several resource-management mechanisms:

- importance/cost selection under bounded memory motivated an exact additive oracle for small/structured experiments;
- H2O (NeurIPS 2023) formulates bounded KV retention as a dynamic submodular problem and motivates a general submodular baseline;
- Quest (ICML 2024) further shows that item utility may be context/query dependent, which is a modelling issue outside the solver itself.

The implementation intentionally stops at reusable mathematical primitives. Attention-specific heavy-hitter scoring, query/key bounds, KV paging, RDMA placement, and serving orchestration remain domain/system mechanisms.

## 5. Deliberately not implemented yet

The current evidence does **not** justify automatically adding all variants of submodular optimization. The following remain future research questions:

- dynamic/streaming submodular maximization;
- non-monotone objectives;
- knapsack-constrained submodular maximization;
- matroid constraints;
- distributed submodular optimization;
- approximate/noisy marginal-gain oracles;
- automatic testing/certification of submodularity for restricted objective classes.

Add these only when independent scientific needs justify them.

## 6. Relationship to ElasticXxx

SciRust is an external scientific R&D platform. ElasticXxx must not depend on SciRust at runtime.

These solvers may be used during experiments to compare policies, build exact small-instance baselines, characterize regret, or investigate objective structure. Any production algorithm selected for ElasticXxx must be implemented autonomously in its target runtime/component.
