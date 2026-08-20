# Exact Budgeted Subset Selection

**Date:** 2026-08-20

## Purpose

`scirust-solvers::budgeted_selection` provides a general, exact and deterministic solver for additive 0/1 selection under one resource budget:

```text
maximize   Σ utility_i x_i
subject to Σ cost_i x_i ≤ budget
           x_i ∈ {0,1}
```

The capability was added while studying importance-aware and multi-tier resource-management mechanisms, but it is intentionally **not** an LLM- or Elastic-specific helper.

## Scientific scope

The primitive is broadly useful for problems including:

- experiment/sample selection under a cost budget;
- cache/object admission studies;
- sensor/feature selection with additive costs;
- task/job subset selection;
- bandwidth/storage allocation experiments;
- tractable exact baselines for heuristic or learned resource policies;
- fixed-point utility optimization in systems research.

## Algorithm

The implementation uses a sparse Pareto-frontier dynamic program rather than a classical table indexed by every budget unit.

After processing each item, a state `(cost_2, utility_2)` is removed when another state `(cost_1, utility_1)` satisfies:

```text
cost_1 <= cost_2
utility_1 >= utility_2
```

For non-negative additive costs and utilities, such a dominated prefix cannot become preferable after adding any subset of the remaining items. Therefore pruning preserves the global optimum.

## Determinism

Inputs use integer `u64` costs and utilities. Floating-point scientific scores should be converted to a documented fixed-point scale before solving.

Tie-breaking is deterministic:

1. maximize total utility;
2. minimize total cost;
3. select the lexicographically smallest vector of input indices.

The returned certificate marks the result as proven optimal and records explored-state and final-frontier counts.

## API

```rust
use scirust_solvers::budgeted_selection::{
    BudgetedItem,
    exact_budgeted_selection,
};

let items = [
    BudgetedItem { cost: 10, utility: 60 },
    BudgetedItem { cost: 20, utility: 100 },
    BudgetedItem { cost: 30, utility: 120 },
];

let solution = exact_budgeted_selection(&items, 50)?;
assert_eq!(solution.selected_indices, vec![1, 2]);
assert_eq!(solution.total_utility, 220);
```

## Relationship to wider optimization tooling

This solver does **not** replace the open investigation into generic LP/ILP/MILP modelling and solving. It covers one important structured 0/1 additive problem exactly, with a simple auditable implementation.

A future generic MIP capability may subsume the mathematical formulation, but this specialized solver can remain valuable as a deterministic pure-Rust exact baseline and as an oracle for validating heuristics on tractable instances.

## Project-boundary rule

This capability belongs in SciRust because it remains scientifically useful if every current consumer project is removed. Runtime-specific actions such as GPU/CPU migration, RDMA transfer, KV-cache lifecycle orchestration, or ElasticXxx transition execution remain outside this module.
